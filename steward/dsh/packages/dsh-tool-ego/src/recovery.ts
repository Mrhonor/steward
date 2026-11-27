/** Run robot fault diagnosis and review in the current Ego Harness session. */
import { randomUUID } from 'node:crypto'
import { setInterval } from 'node:timers/promises'
import type { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { createUserMessage, ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import { defineTool, type ToolDefinition, type ToolExecution } from '@deepseek-ai/dsh-tools'
import type { JsonObject } from '@steward/dsh-ego'

declare module '@deepseek-ai/dsh-llm' {
  interface MessageSourceMap {
    /** Robot evidence for an in-turn task closeout review. */
    'steward-recovery-context': { kind: 'steward-recovery-context' }
  }
}

const OPERATOR_TASK_MARKER = '\u64cd\u4f5c\u5458\u4efb\u52a1\uff1a'
const JSON_OUTPUT = { type: 'json' } as const
const output = { schema: JSON_OUTPUT, render: (_args: object, value: unknown) =>
  [{ type: 'text' as const, text: JSON.stringify(value) }] }
const executionEvidence = 'Compare planned commands with observations from the corresponding execution stage. '
  + 'A Job after snapshot can include retreat or cleanup. Cite Job IDs, execution stages or frame references. '
  + 'If task context already in this conversation and current robot readings leave a gap, state what is unknown.'
const reviewInstructions = 'Check the original fault against the baseline and fresh robot observations. '
  + 'Submit ego_submit_recovery_report with verdict (resolved, unresolved, or insufficient_evidence), reason and '
  + 'bare evidence refs. Cite both baseline and a '
  + 'fresh frame or inspection for a resolved verdict. Use insufficient_evidence when observations cannot settle it. '

interface RecoveryOptions {
  recoveryReasoningEffort: string
  recoveryTimeoutMs: number
}

interface Stage {
  agent: Agent
  id: string
  reviewId?: string
  taskCompletion: boolean
  stop: AbortController
  lease: { owner: string; reviewId?: string }
  supported: boolean
  renewals?: Promise<void>
  timer?: ReturnType<typeof setTimeout>
  disposers: (() => void)[]
  closing?: Promise<void>
}

function reportText(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0 && value.length <= 4096
}

function agentFor(exec: ToolExecution): Agent {
  if (!exec.agent || exec.agent.session.header.parentSession) throw new Error('Recovery requires the current Ego Harness session.')
  return exec.agent
}

/**
 * Install the same-session recovery tools and closeout review hook.
 * @param ctx - the robot tool plugin scope.
 * @param options - recovery-stage reasoning effort and wall-clock budget.
 * @returns the public tools and the handler for motion blocked by mandatory diagnosis.
 */
export function recoveryTools(ctx: Context, options: RecoveryOptions) {
  const ego = ctx.ego
  const stages = new Map<Agent, Stage>()
  const operatorTasks = new WeakMap<Agent, string>()
  const finalText = new WeakMap<Agent['session'], { turn: number; text: string }>()
  const attempted = new WeakMap<Agent, number>()
  let automaticDiagnosis: Promise<JsonObject> | undefined

  async function close(stage: Stage, cancel: boolean): Promise<void> {
    stage.closing ??= (async () => {
      clearTimeout(stage.timer)
      stage.stop.abort()
      try {
        if (cancel) {
          const signal = AbortSignal.timeout(5000)
          if (stage.reviewId) await ego.finishRecoveryReview(stage.id, stage.reviewId, undefined, signal)
          else await ego.finishRecovery(stage.id, false, undefined, signal)
        }
      } catch (error) {
        ctx.logger.warn('The recovery stage could not be closed. Check the fault in ego_state before continuing.', error)
      } finally {
        await stage.renewals
        if (stage.supported) {
          try {
            await ego.recoveryClock(stage.id, { ...stage.lease, operation: 'release' }, AbortSignal.timeout(5000))
          } catch (error) {
            ctx.logger.warn('The recovery clock pause could not be released and will expire.', error)
          }
        }
        for (const dispose of stage.disposers) dispose()
        stages.delete(stage.agent)
      }
    })()
    await stage.closing
  }

  async function open(agent: Agent, id: string, reviewId: string | undefined, taskCompletion: boolean,
    signal: AbortSignal): Promise<void> {
    const stop = new AbortController()
    const lease = { owner: randomUUID(), reviewId }
    const stage: Stage = { agent, id, reviewId, taskCompletion, stop, lease, supported: false, disposers: [] }
    stages.set(agent, stage)
    try {
      const paused = await ego.recoveryClock(id, { ...lease, operation: 'pause' }, signal)
      stage.supported = paused.supported
      if (stage.supported) {
        stage.renewals = (async () => {
          try {
            for await (const _ of setInterval(2000, undefined, { signal: stop.signal, ref: false })) {
              await ego.recoveryClock(id, { ...lease, operation: 'renew' }, stop.signal)
            }
          } catch (error) {
            if (!stop.signal.aborted) ctx.logger.warn('The recovery clock lease renewal failed and will expire.', error)
          }
        })()
      }
      for (const tool of diagnostics) {
        stage.disposers.push(agent.ctx.effect(() => agent.ctx.tools.register(tool)))
      }
      stage.timer = setTimeout(() => {
        ctx.logger.warn('The recovery stage reached its time limit before receiving a report.')
        void close(stage, true)
      }, options.recoveryTimeoutMs)
      stage.timer.unref()
    } catch (error) {
      await close(stage, true)
      throw error
    }
  }

  function active(exec: ToolExecution): { stage: Stage; signal: AbortSignal } {
    const stage = stages.get(agentFor(exec))
    if (!stage || stage.closing) throw new Error('Open a recovery diagnosis or review first.')
    return { stage, signal: AbortSignal.any([exec.signal, stage.stop.signal]) }
  }

  const diagnostics: ToolDefinition[] = [
    defineTool({
      name: 'ego_read_recovery_rgbd',
      description: 'Read a fresh RGB-D robot camera frame without motion. Returns a frame ref, RGB/depth preview, '
        + 'frame intrinsics and camera transforms, timestamp, raw joint encoders and arm geometry.',
      parameters: { camera: { type: 'string', description: 'Robot camera name. Omit for the primary camera.' } },
      output,
      async execute({ camera }, exec) {
        const { stage, signal } = active(exec)
        return JSON.parse(JSON.stringify(await ego.captureRecoveryRgbd(stage.id, { reviewId: stage.reviewId, camera }, signal)))
      },
    }),
    defineTool({
      name: 'ego_analyze_recovery',
      description: 'Analyze captured robot sensor evidence with bounded Python. The input sensor contains body, captureError, '
        + 'kinematics and camera. kinematics includes raw joint encoders, gripper_to_base and armGeometry. For pixel i, '
        + 'u=i%width, v=i//width, camera.rgb[i] is [R,G,B] and camera.depth_m[i] is optical-axis depth in metres, or 0 '
        + 'if missing. Use sampled intrinsics and the matching camera transform. Python supports local assignments, indexing, '
        + 'slices, arithmetic and single-loop list comprehensions. Functions: abs, len, sum, min, max, round, sorted, range, '
        + 'list and mean. Dictionaries support get and keys.',
      parameters: {
        code: { type: 'string', required: true, description: 'Python source with a final JSON result expression.' },
        ref: { type: 'string', description: 'Sensor reference such as baseline, post or a returned frame ref.' },
      },
      output,
      async execute({ code, ref }, exec) {
        const { stage, signal } = active(exec)
        return JSON.parse(JSON.stringify(await ego.analyzeRecovery(stage.id, code, { reviewId: stage.reviewId, ref }, signal)))
      },
    }),
    defineTool({
      name: 'ego_submit_recovery_report',
      description: 'Submit the report for the active recovery stage. Diagnosis requires cause, evidence, recommendation '
        + 'and verification as strings. Review requires verdict (resolved, unresolved, or insufficient_evidence), '
        + 'reason and an array of bare evidence IDs. '
        + 'A task closeout review also requires taskCompleted. Continue based on the returned Runtime state.',
      parameters: { report: { type: 'json', required: true, description: 'Structured diagnosis or review.' } },
      output,
      async execute({ report }, exec) {
        const { stage, signal } = active(exec)
        if (!report || typeof report !== 'object' || Array.isArray(report)) throw new Error('Submit a report object.')
        let result: JsonObject
        if (stage.reviewId) {
          const { verdict, reason, evidence, taskCompleted } = report
          if (verdict !== 'resolved' && verdict !== 'unresolved' && verdict !== 'insufficient_evidence') {
            throw new Error('Use resolved, unresolved, or insufficient_evidence for the review verdict.')
          }
          if (!reportText(reason) || !Array.isArray(evidence) || evidence.length > 16 || !evidence.every(ref => typeof ref === 'string'
              && /^(?:baseline|post|(?:before-frame|frame|inspection|job)-[0-9]+)$/.test(ref))) {
            throw new Error('Submit a reason and bare evidence IDs such as ["baseline","post","frame-2"].')
          }
          if (stage.taskCompletion && typeof taskCompleted !== 'boolean') {
            throw new Error('Submit taskCompleted as a boolean for task closeout.')
          }
          result = await ego.finishRecoveryReview(stage.id, stage.reviewId, { verdict, reason, evidence,
            ...(typeof taskCompleted === 'boolean' ? { taskCompleted } : {}) }, signal)
        } else {
          const { cause, evidence, recommendation, verification } = report
          if (!reportText(cause) || !reportText(evidence) || !reportText(recommendation) || !reportText(verification)) {
            throw new Error('Submit cause, evidence, recommendation and verification as nonempty strings.')
          }
          result = await ego.finishRecovery(stage.id, true, { cause, evidence, recommendation, verification }, signal)
        }
        await close(stage, false)
        return JSON.parse(JSON.stringify(result))
      },
    }),
  ]

  ctx.on('agent/inbox/claimed', ({ agent, message }) => {
    if (message.source.kind !== 'user' || agent.session.header.parentSession) return
    const text = message.content.map(block => block.type === 'text' ? block.text : '').join('\n')
    const marker = text.lastIndexOf(OPERATOR_TASK_MARKER)
    if (marker < 0) return
    const task = text.slice(marker + OPERATOR_TASK_MARKER.length).trim()
    if (task) operatorTasks.set(agent, task.slice(0, 4000))
  })
  ctx.on('agent/request', async ({ agent }, next) => {
    const config = await next()
    return stages.has(agent) ? { ...config, reasoningEffort: ReasoningEffortId(options.recoveryReasoningEffort) } : config
  }, { prepend: true })
  ctx.effect(() => ctx.tools.guard(exec => {
    if (exec.agent && stages.has(exec.agent) && (exec.name === 'skill_run' || exec.name === 'ego_explore')) {
      return 'Submit the active recovery report, or abandon recovery, before issuing motion.'
    }
  }))
  ctx.on('agent/status', ({ agent, status }) => {
    const stage = stages.get(agent)
    if (status === 'idle' && stage) void close(stage, true)
  })
  ctx.effect(() => async () => {
    await Promise.all([...stages.values()].map(stage => close(stage, true)))
  })
  ctx.on('session/event', (session, event) => {
    if (session.header.parentSession || event.type !== 'assistant/message') return
    const { turn, message, interrupted } = event.data
    if (interrupted || message.content.some(block => block.type === 'tool-call')) {
      finalText.delete(session)
      return
    }
    const text = message.content.map(block => block.type === 'text' ? block.text : '').join('\n').trim()
    if (text) finalText.set(session, { turn, text: text.slice(0, 4000) })
  })
  ctx.on('agent/turn-stopping', async ({ agent, turn, signal }) => {
    const stage = stages.get(agent)
    if (stage) {
      await close(stage, true)
      return
    }
    const task = operatorTasks.get(agent)
    const claim = finalText.get(agent.session)
    if (!task || claim?.turn !== turn || attempted.get(agent) === turn || signal.aborted) return
    const callSignal = AbortSignal.any([signal, AbortSignal.timeout(options.recoveryTimeoutMs)])
    try {
      if ((await ego.faultState(callSignal)).state === 'NORMAL') return
      attempted.set(agent, turn)
      const opened = await ego.beginTaskRecoveryReview(task, claim.text, callSignal)
      if (!('reviewId' in opened)) return
      await open(agent, opened.id, opened.reviewId, true, callSignal)
      agent.steer(createUserMessage({ source: { kind: 'steward-recovery-context' }, content: [{ type: 'text',
        text: 'Review the operator task and fault before ending this turn. Treat your draft conclusion as a claim to check. '
          + reviewInstructions + 'Also report taskCompleted as a boolean. After the report tool returns, describe the actual '
          + 'task and recovery state.\n\n' + JSON.stringify(opened.evidence),
      }] }))
    } catch (error) {
      if (!signal.aborted) ctx.logger.warn('The task closeout review could not open. The fault status is unverified.', error)
    }
  })

  async function diagnose(agent: Agent, context: string, job: string | undefined, signal: AbortSignal): Promise<JsonObject> {
    const opened = await ego.beginRecovery(context, job, signal)
    await open(agent, opened.id, undefined, false, signal)
    return { id: opened.id, state: 'RECOVERY', phase: 'diagnosing',
      instruction: 'Diagnose this fault in the current conversation. Use the recovery RGB-D and Python tools if needed. '
        + executionEvidence + 'Submit ego_submit_recovery_report with cause, evidence, recommendation and verification '
        + 'before attempting corrective motion. Recovery sensors and Python are available during diagnosis and review. '
        + 'Use regular observation tools between those stages.',
      diagnostic: opened.diagnostic }
  }

  async function diagnoseBlockedMotion(agent: Agent, signal: AbortSignal): Promise<JsonObject> {
    const state = await ego.faultState(signal)
    if (!state.diagnosisRequired) return { ...state }
    if (automaticDiagnosis) return automaticDiagnosis
    if (state.recovery) return { ...state, reason: 'Recovery is already active.' }
    automaticDiagnosis = diagnose(agent, 'A command was blocked before movement because a fault action repeated while DEGRADED. '
      + 'Find the cause before choosing corrective action.', state.diagnosisRequired.job, signal)
    try {
      return await automaticDiagnosis
    } catch (error) {
      signal.throwIfAborted()
      ctx.logger.warn('The mandatory diagnosis could not open.', error)
      return { ...state, reason: 'Motion remains blocked. Check the fault in ego_state and retry with ego_begin_recovery.' }
    } finally {
      automaticDiagnosis = undefined
    }
  }

  const tools: ToolDefinition[] = [
    defineTool({
      name: 'ego_begin_recovery',
      description: 'Enter RECOVERY from DEGRADED. Return the fault history and open diagnostic tools in this conversation.',
      parameters: {
        context: { type: 'string', required: true, description: 'Task goal, constraints, earlier attempts, evidence and uncertainty.' },
        job: { type: 'string', description: 'Optional recorded fault Job ID.' },
      },
      output,
      async execute({ context, job }, exec) {
        return JSON.parse(JSON.stringify(await diagnose(agentFor(exec), context, job, exec.signal)))
      },
    }),
    defineTool({
      name: 'ego_verify_recovery',
      description: 'After corrective action, open a review with fresh robot evidence. Submit the verdict in this conversation.',
      parameters: {
        id: { type: 'string', required: true, description: 'Active recovery attempt ID.' },
        referenceBacks: { type: 'array', items: { type: 'string' }, description: 'Optional Ego-State IDs of cardboard reference pieces.' },
      },
      output,
      async execute({ id, referenceBacks }, exec) {
        const agent = agentFor(exec)
        const opened = await ego.beginRecoveryReview(id, referenceBacks ?? [], exec.signal)
        if (!('reviewId' in opened)) return JSON.parse(JSON.stringify(opened))
        await open(agent, id, opened.reviewId, false, exec.signal)
        return { id, state: 'RECOVERY', phase: 'reviewing', reviewId: opened.reviewId,
          instruction: reviewInstructions + executionEvidence, evidence: opened.evidence }
      },
    }),
    defineTool({
      name: 'ego_abandon_recovery',
      description: 'Return an unverified recovery attempt to DEGRADED.',
      parameters: { id: { type: 'string', required: true, description: 'Active recovery attempt ID.' } },
      output,
      async execute({ id }, exec) {
        const stage = stages.get(agentFor(exec))
        const result = await ego.finishRecovery(id, false, undefined, exec.signal)
        if (stage?.id === id) await close(stage, false)
        return JSON.parse(JSON.stringify(result))
      },
    }),
  ]
  return { tools, diagnoseBlockedMotion }
}
