// Replay fault reporting and restricted recovery conversations through the Ego profile.
import assert from 'node:assert/strict'
import { once } from 'node:events'
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { test } from 'node:test'
import { pathToFileURL } from 'node:url'
import { launchProfile } from './profile-test.mjs'

const cliPackage = createRequire(import.meta.url).resolve('@deepseek-ai/dsh/package.json')
const readRows = async file => (await readFile(file, 'utf8')).trim().split('\n').map(line => JSON.parse(line))
const userTask = rows => rows.find(row => row.type === 'user/message' && row.data.source.kind === 'user').data.content[0].text
const toolValue = row => JSON.parse(row.data.content[0].text)

for (const name of ['fault-monitor', 'recovery', 'recovery-context', 'recovery-closeout', 'mandatory-diagnosis',
  'proactive-degradation']) test(`Ego Harness replays ${name} without a model key`,
  { timeout: 240_000 }, async t => {
  const directory = join(import.meta.dirname, 'snapshots', name)
  const session = join(directory, 'session.v4.jsonl')
  const childSession = join(directory, 'session.1.v4.jsonl')
  const rows = await readRows(session)
  const child = name === 'fault-monitor' ? null : await readRows(childSession)
  const closeout = name === 'recovery-closeout'
  const mandatory = name === 'mandatory-diagnosis'
  const proactive = name === 'proactive-degradation'
  const promptBlocks = child?.filter(row => row.type === 'user/message').flatMap(row => row.data.content) ?? []
  const evidenceOf = text => JSON.parse(text.slice(text.indexOf('\n\n') + 2))
  const reviewText = promptBlocks.find(block => block.text?.startsWith(closeout || mandatory || proactive
    ? 'Review the operator task' : 'Review the recovery'))?.text
  const reviewEvidence = reviewText ? evidenceOf(reviewText) : {}
  delete reviewEvidence.cameraFault
  delete reviewEvidence.correctiveAction
  const answers = promptBlocks.filter(block => block.text?.startsWith('EH answered your context request'))
    .map(block => evidenceOf(block.text))
  const prompt = userTask(rows)
  const dispatches = rows.filter(row => row.type === 'tool/ptc-dispatch')
  const resultFor = tool => toolValue(dispatches.findLast(row => row.data.name === tool))
  const predictions = proactive ? dispatches.filter(row => row.data.name === 'ego_predict').map(toolValue) : []
  const picks = proactive ? dispatches.filter(row => row.data.name === 'skill_run').map((row) => ({
    receipt: row.data.arguments.receipt, ...toolValue(row),
  })) : []
  const exchanges = ['ego_begin_recovery', 'ego_answer_recovery_context']
  const pending = dispatches.filter(row => exchanges.includes(row.data.name)).map(toolValue).filter(value => value.needsContext)
  const recordedReports = child?.filter(row => row.type === 'tool/call' && row.data.name === 'ego_submit_recovery_report') ?? []
  const reviewTurn = recordedReports.at(-1)?.data.turn
  const childTools = { ego_analyze_recovery: '/ego/recovery/analyze', ego_read_recovery_rgbd: '/ego/recovery/rgbd' }
  const reads = child?.filter(row => row.type === 'tool/call' && Object.hasOwn(childTools, row.data.name))
    .map(call => ({ path: childTools[call.data.name], review: call.data.turn === reviewTurn,
      args: JSON.parse(call.data.arguments),
      result: child.find(row => row.type === 'tool/result' && row.sourceEventSeqs.includes(call.seq)).data.message.content[0].text })) ?? []
  let readIndex = 0
  let requestIndex = 0
  let answerIndex = 0
  let childId
  let parentId
  let scanCount = 0
  let pickPredictions = 0
  let picksStarted = 0
  const root = await mkdtemp(join(tmpdir(), 'dsh-fault-'))
  const calls = []
  const clockPauses = []
  let lease
  let state = { state: 'NORMAL' }
  const server = createServer(async (req, res) => {
    const chunks = []
    for await (const chunk of req) chunks.push(chunk)
    const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : null
    res.setHeader('content-type', 'application/json')
    if (req.method === 'POST' && req.url === '/ego/recovery/clock') {
      assert.ok(child)
      const { operation, ...owner } = body
      if (operation === 'pause') {
        assert.equal(lease, undefined)
        assert.ok(owner.owner)
        clockPauses.push(owner.owner)
        lease = owner
      } else {
        assert.deepEqual(owner, lease)
        if (operation === 'release') lease = undefined
        else assert.equal(operation, 'renew')
      }
      res.end(JSON.stringify({ supported: true }))
      return
    }
    calls.push([req.method, req.url, body])
    if (req.method === 'GET' && req.url === '/ego/summary') {
      res.end(JSON.stringify({ ...(child ? resultFor('ego_state') : {}), fault: state }))
    } else if (req.method === 'GET' && req.url === '/ego/fault/state') {
      res.end(JSON.stringify(state))
    } else if (proactive && req.method === 'POST' && req.url === '/ego/predict') {
      const prediction = predictions[pickPredictions++]
      res.end(JSON.stringify({ ...prediction, id: prediction.receipt }))
    } else if (proactive && req.method === 'GET' && req.url?.startsWith('/ego/receipt?')) {
      const receipt = new URLSearchParams(req.url.split('?')[1]).get('id')
      const prediction = predictions.find(p => p.receipt === receipt)
      res.end(JSON.stringify({ ...prediction, id: receipt }))
    } else if (proactive && req.method === 'POST' && req.url === '/skill/run') {
      const job = picks[picksStarted++]
      assert.equal(body.receipt, job.receipt)
      assert.equal(state.state, picksStarted === 1 ? 'NORMAL' : 'RECOVERY')
      res.end(JSON.stringify({ ...job, id: job.job, state: 'running' }))
    } else if (proactive && req.method === 'POST' && req.url === '/skill/wait') {
      const job = picks.find(job => job.job === body.job)
      res.end(JSON.stringify({ ...job, id: job.job }))
    } else if (proactive && req.method === 'POST' && req.url === '/ego/observe') {
      res.end(JSON.stringify(resultFor('ego_observe')))
    } else if (proactive && req.method === 'GET' && req.url === '/ego/gripper/pose') {
      res.end(JSON.stringify(resultFor('ego_gripper_pose')))
    } else if (proactive && req.method === 'POST' && req.url === '/ego/alert') {
      res.end(JSON.stringify(resultFor('report_problem')))
    } else if (req.method === 'POST' && req.url === '/ego/fault/degrade') {
      if (proactive) {
        assert.equal(picksStarted, 1)
        assert.match(body.reason, /job-1.*e:1.*7 grasp probes.*empty gripper/)
      }
      state = { state: 'DEGRADED', transition: { stamp: 0, code: 'agent_report', reason: body.reason } }
      res.end(JSON.stringify(state))
    } else if (mandatory && req.method === 'POST' && req.url === '/ego/explore') {
      const first = toolValue(dispatches.find(row => row.data.name === 'ego_explore'))
      if (++scanCount === 1) res.end(JSON.stringify({ ...first, id: first.job, state: 'running' }))
      else {
        state = { ...state, diagnosisRequired: { job: first.job, operation: 'look_around', target: null } }
        res.writeHead(409).end(JSON.stringify({ error: 'diagnosis required', code: 'diagnosis_required' }))
      }
    } else if (mandatory && req.method === 'POST' && req.url === '/skill/wait') {
      const first = toolValue(dispatches.find(row => row.data.name === 'ego_explore'))
      assert.equal(body.job, first.job)
      res.end(JSON.stringify({ ...first, id: first.job }))
    } else if (child && req.method === 'POST' && req.url === '/ego/reflect') {
      res.end(JSON.stringify({ cause: 'unknown', sought: null, targets: [], missing: [], candidates: [] }))
    } else if (child && req.method === 'POST' && req.url === '/ego/recovery/start') {
      const message = userTask(child)
      const diagnostic = evidenceOf(message)
      if (mandatory) assert.equal(body.job, state.diagnosisRequired.job)
      if (proactive) {
        assert.equal(body.job, picks[0].job)
        assert.equal(state.state, 'DEGRADED')
      }
      res.end(JSON.stringify({ id: diagnostic.fault.recovery.id, diagnostic }))
    } else if (child && req.method === 'POST' && reads[readIndex]?.path === req.url) {
      assert.equal(lease?.id, body.id)
      assert.equal(lease.reviewId, body.reviewId)
      const expected = reads[readIndex++]
      assert.deepEqual({ code: body.code, ref: body.ref, camera: body.camera },
        { code: expected.args.code, ref: expected.args.ref, camera: expected.args.camera })
      assert.equal(body.reviewId, expected.review ? 'review-2' : undefined)
      const value = expected.result
      if (value.startsWith('Error: ')) res.writeHead(409).end(JSON.stringify({ error: value.slice(7) }))
      else res.end(value)
    } else if (child && req.method === 'POST' && req.url === '/ego/recovery/context/request') {
      assert.equal(lease?.id, body.id)
      const recorded = pending[requestIndex++]
      childId ??= body.childId
      parentId ??= body.parentId
      assert.equal(body.childId, childId)
      assert.equal(body.parentId, parentId)
      assert.deepEqual({ question: body.question, reason: body.reason },
        { question: recorded.request.question, reason: recorded.request.reason })
      res.end(JSON.stringify({ ...recorded, recovery: { ...recorded.recovery, childId } }))
    } else if (child && req.method === 'POST' && req.url === '/ego/recovery/context/answer') {
      assert.equal(lease, undefined)
      const evidence = answers[answerIndex++]
      assert.equal(body.parentId, parentId)
      assert.equal(body.requestId, evidence.request.id)
      assert.equal(body.answer, evidence.answer)
      res.end(JSON.stringify({ childId, reviewId: null, evidence }))
    } else if (child && req.method === 'POST' && req.url === '/ego/recovery/finish' && body.completed === true) {
      assert.equal(lease?.id, body.id)
      const { diagnosis, ...status } = mandatory ? resultFor('ego_explore').diagnostic
        : resultFor(answers.length ? 'ego_answer_recovery_context' : 'ego_begin_recovery')
      childId ??= body.childId
      assert.equal(body.childId, childId)
      assert.match(childId, /^session-steward-recovery-/)
      assert.deepEqual(body.report, diagnosis)
      state = { ...status, recovery: { ...status.recovery, childId } }
      res.end(JSON.stringify(state))
    } else if (child && req.method === 'POST' && req.url === '/ego/recovery/verify') {
      if (body.id !== state.recovery.id) {
        res.writeHead(409).end(JSON.stringify({ error: 'recovery is no longer active in the requested phase' }))
        return
      }
      res.end(JSON.stringify({ reviewId: 'review-2', childId, evidence: reviewEvidence }))
    } else if ((closeout || mandatory || proactive) && req.method === 'POST' && req.url === '/ego/recovery/task-review') {
      assert.equal(state.state, closeout ? 'DEGRADED' : 'RECOVERY')
      assert.deepEqual({ task: body.task, claim: body.claim }, reviewEvidence.taskCompletion)
      if (closeout) childId = body.childId
      assert.match(childId, /^session-steward-recovery-/)
      state = { state: 'RECOVERY' }
      res.end(JSON.stringify({ id: 'recovery-1', reviewId: 'review-2', childId, newChild: closeout, evidence: reviewEvidence }))
    } else if (!child && req.method === 'POST' && req.url === '/ego/recovery/task-review') {
      res.end(JSON.stringify({ ...state, reason: 'wait for the active skill to finish' }))
    } else if (child && req.method === 'POST' && req.url === '/ego/recovery/verify/finish') {
      assert.equal(lease?.id, body.id)
      assert.equal(lease.reviewId, body.reviewId)
      assert.equal(body.reviewId, 'review-2')
      const result = proactive ? { state: 'RECOVERY', verified: false,
        review: JSON.parse(recordedReports.at(-1).data.arguments).report } : closeout || mandatory ? { state: 'NORMAL', verified: true, reason: '',
        review: JSON.parse(recordedReports.at(-1).data.arguments).report } : resultFor('ego_verify_recovery')
      assert.deepEqual(body.report, result.review)
      state = { state: result.state }
      res.end(JSON.stringify(result))
    } else {
      res.writeHead(404).end(JSON.stringify({ error: 'Unexpected Ego request' }))
    }
  })
  t.after(async () => {
    server.closeAllConnections()
    await new Promise(done => server.close(done))
    await rm(root, { recursive: true, force: true })
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')

  const env = { ...process.env, DSH_HOME: join(root, 'home'), DSH_SNAPSHOT_FILE: session }
  if (child) env.DSH_SNAPSHOT_CHILD_FILES = childSession
  delete env.DEEPSEEK_API_KEY
  delete env.DEEPSEEK_BASE_URL
  const launch = args => launchProfile(args, root, env)
  await launch(['--profile', 'ego', '--from-default-profile', 'headless', '--dump-default-config'])
  await launch(['plugin', '--profile', 'ego', 'add', resolve(import.meta.dirname, '../packages/dsh-bundle-ego')])
  const patch = join(root, 'replay.json')
  await writeFile(patch, JSON.stringify([
    { id: 'steward-ego-runtime', config: { endpoint: `http://127.0.0.1:${server.address().port}` } },
    { id: 'session-title-llm', disabled: true },
    { id: 'session-persistence-jsonl', config: { root: join(root, 'sessions'), compression: 'none' } },
    { id: 'session-log-deepseek', config: { enabled: false } },
    { id: 'llm-deepseek', disabled: true },
    { id: 'llm-deepseek-account', disabled: true },
    { id: 'agent-default-model', config: { provider: 'deepseek-official', model: 'deepseek-flash' } },
    { insert: [{ id: 'llm-replay', name: pathToFileURL(createRequire(cliPackage).resolve('@deepseek-ai/dsh-llm-replay')).href,
      config: { providers: [{ id: 'deepseek-official',
        models: [{ id: 'deepseek-flash', contextWindow: 128000, reasoningEfforts: ['low', 'medium', 'high', 'max'] }] }] } }] },
  ]))
  const output = await launch(['--profile', 'ego', '--patch', patch, '--json', prompt])
  const events = output.trim().split('\n').filter(line => line.startsWith('{')).map(line => JSON.parse(line))
  const results = events.filter(event => event.type === 'tool_result')
  assert.equal(lease, undefined)
  assert.equal(new Set(clockPauses).size, recordedReports.length + pending.length)
  if (child) {
    assert.ok(results.some(result => result.status === 'completed'), 'the enclosing EH tool call completed')
    assert.equal(calls.findLast(call => call[1] === '/ego/recovery/verify/finish')?.[2].report.verdict,
      proactive ? 'unresolved' : 'resolved')
    assert.equal(readIndex, reads.length)
    assert.equal(requestIndex, pending.length)
    assert.equal(answerIndex, answers.length)
    if (proactive) {
      const paths = calls.map(call => call[1])
      const degrade = paths.indexOf('/ego/fault/degrade')
      assert.ok(paths.indexOf('/skill/wait') < degrade && degrade < paths.indexOf('/ego/recovery/start'))
      assert.ok(paths.indexOf('/ego/recovery/finish') < paths.lastIndexOf('/skill/run'))
      assert.equal(picksStarted, 2)
      assert.equal(state.state, 'RECOVERY')
    } else assert.deepEqual(calls.map(call => call[1]), mandatory ? [
      '/ego/fault/degrade', '/ego/explore', '/skill/wait', '/ego/explore', '/ego/fault/state', '/ego/recovery/start',
      ...reads.filter(read => !read.review).map(read => read.path), '/ego/recovery/finish',
      '/ego/summary', '/ego/fault/state', '/ego/recovery/task-review',
      ...reads.filter(read => read.review).map(read => read.path), '/ego/recovery/verify/finish',
    ] : closeout ? [
      '/ego/fault/degrade', '/ego/summary', '/ego/fault/state', '/ego/recovery/task-review',
      ...reads.map(read => read.path), '/ego/recovery/verify/finish',
    ] : [
      '/ego/fault/degrade', '/ego/recovery/start',
      ...reads.filter(read => !read.review).map(read => read.path),
      ...answers.flatMap(() => ['/ego/recovery/context/request', '/ego/recovery/context/answer']),
      '/ego/recovery/finish', '/ego/recovery/verify',
      ...(answers.length ? [] : ['/ego/reflect', '/ego/recovery/verify']),
      ...reads.filter(read => read.review).map(read => read.path),
      '/ego/recovery/verify/finish',
      ...(answers.length ? ['/ego/fault/state'] : []),
    ])
    const files = await readdir(join(root, 'sessions'), { recursive: true })
    const logs = files.filter(file => file.includes('session-steward-recovery-') && file.endsWith('session.v4.jsonl'))
    assert.equal(logs.length, 1)
    const childRows = await readRows(join(root, 'sessions', logs[0]))
    assert.equal(childRows[0].id, childId)
    const reports = childRows.filter(row => row.type === 'tool/call' && row.data.name === 'ego_submit_recovery_report')
    assert.deepEqual(reports.map(row => row.data.turn), recordedReports.map(row => row.data.turn))
    if (proactive) {
      const { actions, sensor } = evidenceOf(userTask(childRows))
      const pick = actions.find(action => action.job === 'job-1')
      assert.ok(pick.execution.log.includes('docked'))
      const { stages } = pick.execution
      assert.deepEqual(stages.map(stage => stage.phase),
        ['dock_verified', ...Array(7).fill('grasp_probe_feedback'), 'backoff_started', 'backoff_finished'])
      assert.ok(stages[0].feedback.odometry.x > 0)
      assert.ok(stages.at(-1).feedback.odometry.x < 0)
      assert.deepEqual(stages.at(-1).targetXY, pick.plan.predock)
      const { camera, kinematics } = sensor
      assert.equal(camera.stamp, kinematics.stamp)
      assert.equal(camera.camera_axes, 'opencv_x_right_y_down_z_forward')
      assert.equal(camera.K.length, 3)
      assert.equal(camera.camera_to_base.length, 4)
      assert.equal(camera.camera_to_odometry.length, 4)
      assert.ok(kinematics.armGeometry.toolOffsetM > 0)
      assert.equal(kinematics.gripper_to_base.length, 4)
    }
    if (closeout) assert.equal(JSON.parse(reports[0].data.arguments).report.taskCompleted, true)
    if (mandatory) {
      assert.equal(scanCount, 2)
      assert.ok(results.some(result => typeof result.result === 'string' && result.result.includes('diagnosis_required')))
      assert.equal(JSON.parse(reports[1].data.arguments).report.taskCompleted, true)
    }
    if (name === 'recovery-context') {
      const questions = childRows.filter(row => row.type === 'tool/call' && row.data.name === 'ego_request_recovery_context')
      assert.deepEqual(questions.map(row => row.data.turn), [1, 2])
      assert.match(evidenceOf(userTask(childRows)).operatorTasks[0], /^Run a stationary diagnostic conversation test\./)
      assert.deepEqual(reports.map(row => row.data.turn), [3, 4])
    }
    const requests = childRows.filter(row => row.type === 'request/header')
    assert.ok(requests.length > 0)
    for (const row of requests) {
      assert.equal(row.data.header.config.reasoningEffort, 'max')
      assert.deepEqual(row.data.header.tools.map(tool => tool.name).sort(),
        ['ego_analyze_recovery', 'ego_read_recovery_rgbd', 'ego_request_recovery_context', 'ego_submit_recovery_report'])
    }
    assert.ok(events.some(event => event.type === 'final'))
    return
  }
  assert.equal(results.length, 1, output)
  assert.equal(results[0].status, 'completed', output)
  const expected = rows.find(row => row.type === 'tool/result').data.message.content[0].text
  const faultLines = text => text.split('\n').filter(line => /^(FIRST_FAULT|REPORT|SECOND_FAULT)=/.test(line))
  assert.deepEqual(faultLines(results[0].result), faultLines(expected))
  assert.deepEqual(calls.slice(0, 3), [
    ['GET', '/ego/summary', null],
    ['POST', '/ego/fault/degrade', { reason: '观察到重复操作，需记录' }],
    ['GET', '/ego/summary', null],
  ])
  assert.deepEqual(calls.slice(3).map(call => call.slice(0, 2)), [
    ['GET', '/ego/fault/state'],
    ['POST', '/ego/recovery/task-review'],
  ])
  assert.equal(state.state, 'DEGRADED')
  assert.ok(events.some(event => event.type === 'final'))
})
