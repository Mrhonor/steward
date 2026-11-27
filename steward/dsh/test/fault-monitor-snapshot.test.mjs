// Replay the Ego profile against a synthetic robot endpoint.
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
const taskSource = 'steward-recovery-context'
const recoveryId = 'recovery-1'
const reviewId = 'review-2'

for (const name of ['fault-monitor', 'recovery', 'recovery-closeout', 'mandatory-diagnosis', 'proactive-degradation',
  'recovery-interrupted']) test(`Ego Harness replays ${name} without a model key`, { timeout: 240_000 }, async t => {
  const snapshot = join(import.meta.dirname, 'snapshots', name, 'session.v4.jsonl')
  const rows = await readRows(snapshot)
  const prompt = rows.find(row => row.type === 'user/message' && row.data.source.kind === 'user').data.content[0].text
  const faultOnly = name === 'fault-monitor'
  const mandatory = name === 'mandatory-diagnosis'
  const proactive = name === 'proactive-degradation'
  const explicit = name === 'recovery'
  const cancelled = name === 'recovery-interrupted'
  const closeout = mandatory || proactive || name === 'recovery-closeout'

  const root = await mkdtemp(join(tmpdir(), 'dsh-fault-'))
  const calls = []
  const pauses = []
  let lease
  let state = { state: 'NORMAL' }
  let scans = 0
  let picks = 0
  let predictions = 0
  let diagnosis
  let report
  const packet = { ref: 'baseline', camera: { name: 'head_camera', K: [[50, 0, 32], [0, 50, 24], [0, 0, 1]] } }
  const job = (id, skill, receipt) => ({ id, state: 'succeeded', skill, receipt, log: [], message: 'finished' })
  const faultJob = {
    job: 'job-1', operation: proactive ? 'pick' : 'look_around',
    execution: { log: ['docked'], stages: [{ phase: 'dock_verified', stamp: 0.1, feedback: { gripper: { holding: false } } }] },
  }
  const makeRecovery = phase => ({ ...state, state: 'RECOVERY', recovery: { id: recoveryId, phase } })
  const receipt = id => ({ id, skill: 'pick', params: { target: 'e:1' }, feasible: true, issues: [], risk: 0.05,
    expiresAt: 100, plan: { object: 'e:1' } })
  const errors = []
  const server = createServer(async (req, res) => {
    try {
      const chunks = []
      for await (const chunk of req) chunks.push(chunk)
      const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : null
      const url = new URL(req.url, 'http://localhost')
      const path = url.pathname
      const reply = (value, status = 200) => res.writeHead(status, { 'content-type': 'application/json' }).end(JSON.stringify(value))
      if (path === '/ego/control') return reply({ owner: null })
      calls.push(path)
      switch (path) {
        case '/ego/summary':
          return reply({ fault: state })
        case '/ego/fault/state':
          return reply(state)
        case '/ego/fault/degrade':
          state = { state: 'DEGRADED', transition: { stamp: 0, code: 'agent_report', reason: body.reason } }
          return reply(state)
        case '/ego/explore':
          assert.ok(mandatory, 'diagnostic tools may not dispatch motion')
          if (++scans === 1) return reply(job('job-1', 'look_around', 'look-around'))
          state.diagnosisRequired = { job: 'job-1', operation: 'look_around', target: null }
          return reply({ code: 'diagnosis_required', error: 'Repeated scan requires diagnosis' }, 409)
        case '/ego/predict':
          assert.ok(proactive)
          assert.equal(body.params.target, 'e:1')
          return reply(receipt(`rcpt-${++predictions}`))
        case '/ego/receipt':
          return reply(receipt(url.searchParams.get('id')))
        case '/skill/run':
          assert.ok(proactive)
          assert.equal(state.state, ++picks === 1 ? 'NORMAL' : 'RECOVERY')
          assert.equal(lease, undefined)
          return reply(job(`job-${picks}`, 'pick', body.receipt))
        case '/skill/wait':
          if (mandatory) return reply(job(body.job, 'look_around', 'look-around'))
          return reply({ ...job(body.job, 'pick', `rcpt-${picks}`), state: 'failed', message: 'grasp missed e:1' })
        case '/ego/observe':
          assert.equal(lease, undefined)
          return reply({ objects: [{ id: 'e:1', category: 'mug' }] })
        case '/ego/reflect':
          return reply({ cause: 'unknown', sought: null, targets: [], missing: [], candidates: [] })
        case '/ego/recovery/start':
          assert.equal(state.state, 'DEGRADED')
          assert.equal(body.job, 'job-1')
          state = makeRecovery('diagnosing')
          return reply({ id: recoveryId, diagnostic: { baseline: packet, faultJob, actions: [faultJob] } })
        case '/ego/recovery/clock': {
          const { operation, ...owner } = body
          assert.equal(owner.id, recoveryId)
          if (operation === 'pause') {
            assert.equal(lease, undefined)
            assert.ok(owner.owner)
            pauses.push(owner)
            lease = owner
          } else {
            assert.deepEqual(owner, lease)
            if (operation === 'release') lease = undefined
            else assert.equal(operation, 'renew')
          }
          return reply({ supported: true })
        }
        case '/ego/recovery/rgbd':
        case '/ego/recovery/analyze':
          assert.equal(lease?.id, body.id)
          assert.equal(lease.reviewId, body.reviewId)
          if (path.endsWith('rgbd')) {
            assert.equal(body.camera, 'head_camera')
            return reply({ ...packet, ref: body.reviewId ? 'frame-3' : 'before-frame-1' })
          }
          assert.equal(body.code, "sensor['camera']['depth_m'][0]")
          assert.equal(body.ref, body.reviewId ? 'frame-3' : 'before-frame-1')
          return reply({ value: body.reviewId ? 0.8 : 0 })
        case '/ego/recovery/finish':
          assert.equal(lease?.id, body.id)
          assert.equal(state.recovery.phase, 'diagnosing')
          if (body.completed) {
            assert.deepEqual(Object.keys(body.report).sort(), ['cause', 'evidence', 'recommendation', 'verification'])
            diagnosis = body.report
            delete state.diagnosisRequired
            state = makeRecovery('verifying')
          } else {
            assert.ok(cancelled)
            state = { state: 'DEGRADED', transition: state.transition }
          }
          return reply(state)
        case '/ego/recovery/verify':
        case '/ego/recovery/task-review': {
          assert.equal(lease, undefined)
          const taskCompletion = path.endsWith('task-review') ? { task: body.task, claim: body.claim } : undefined
          if (faultOnly) return reply({ ...state, reason: 'wait for the active skill to finish' })
          if (taskCompletion) {
            assert.ok(closeout)
            assert.equal(body.task, prompt.split('\uff1a').at(-1))
            assert.ok(body.claim)
            if (mandatory || proactive) assert.equal(state.recovery.phase, 'verifying')
          } else {
            assert.ok(explicit)
            assert.equal(state.recovery.phase, 'verifying')
            assert.equal(body.id, recoveryId)
          }
          state = makeRecovery('reviewing')
          return reply({ id: recoveryId, reviewId, evidence: { taskCompletion, diagnosis, baseline: packet, post: { ref: 'post' } } })
        }
        case '/ego/recovery/verify/finish':
          assert.equal(lease?.id, body.id)
          assert.equal(lease.reviewId, body.reviewId)
          report = body.report
          assert.deepEqual(report.evidence, ['baseline', 'post'])
          assert.equal(report.verdict, proactive ? 'unresolved' : 'resolved')
          assert.equal(report.taskCompleted, closeout ? !proactive : undefined)
          state = proactive ? makeRecovery('verifying') : { state: 'NORMAL' }
          return reply({ ...state, verified: !proactive, review: report })
        default:
          throw new Error(`Unexpected Ego request ${path}`)
      }
    } catch (error) {
      errors.push(error)
      res.writeHead(500).end(JSON.stringify({ error: 'Ego fixture assertion failed' }))
    }
  })
  t.after(async () => {
    server.closeAllConnections()
    await new Promise(done => server.close(done))
    await rm(root, { recursive: true, force: true })
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')

  const env = { ...process.env, DSH_HOME: join(root, 'home'), DSH_SNAPSHOT_FILE: snapshot }
  delete env.DSH_SNAPSHOT_CHILD_FILES
  delete env.DEEPSEEK_API_KEY
  delete env.DEEPSEEK_BASE_URL
  const launch = args => launchProfile(args, root, env)
  await launch(['--profile', 'ego', '--from-default-profile', 'headless', '--dump-default-config'])
  await launch(['plugin', '--profile', 'ego', 'add', resolve(import.meta.dirname, '../packages/dsh-bundle-ego')])
  const patch = join(root, 'replay.json')
  const sessions = join(root, 'sessions')
  await writeFile(patch, JSON.stringify([
    { id: 'steward-ego-runtime', config: { endpoint: `http://127.0.0.1:${server.address().port}` } },
    { id: 'session-title-llm', disabled: true },
    { id: 'session-persistence-jsonl', config: { root: sessions, compression: 'none' } },
    { id: 'session-log-deepseek', config: { enabled: false } },
    { id: 'llm-deepseek', disabled: true },
    { id: 'llm-deepseek-account', disabled: true },
    { id: 'agent-default-model', config: { provider: 'deepseek-official', model: 'deepseek-flash' } },
    { insert: [{ id: 'llm-replay', name: pathToFileURL(createRequire(cliPackage).resolve('@deepseek-ai/dsh-llm-replay')).href,
      config: { providers: [{ id: 'deepseek-official',
        models: [{ id: 'deepseek-flash', contextWindow: 128000, reasoningEfforts: ['low', 'medium', 'high', 'max'] }] }] } }] },
  ]))
  const output = await launch(['--profile', 'ego', '--patch', patch, '--json', prompt])
  assert.deepEqual(errors, [])
  assert.equal(lease, undefined)
  assert.equal(pauses.length, faultOnly ? 0 : cancelled || name === 'recovery-closeout' ? 1 : 2)
  assert.equal(new Set(pauses.map(pause => pause.owner)).size, pauses.length)
  assert.equal(state.state, faultOnly || cancelled ? 'DEGRADED' : proactive ? 'RECOVERY' : 'NORMAL')
  const files = await readdir(sessions, { recursive: true })
  const logs = files.filter(file => file.endsWith('session.v4.jsonl'))
  assert.equal(logs.length, 1, 'all recovery activity belongs to the EH session')
  const log = await readRows(join(sessions, logs[0]))
  assert.equal(log[0].parentSession, undefined)
  assert.equal(log.filter(row => row.type === 'user/message' && row.data.source.kind === taskSource).length, closeout ? 1 : 0)
  const dispatches = log.filter(row => row.type === 'tool/ptc-dispatch')
  const reports = dispatches.filter(row => row.data.name === 'ego_submit_recovery_report')
  assert.equal(reports.length, faultOnly || cancelled ? 0 : name === 'recovery-closeout' ? 1 : 2)
  assert.ok(reports.every(row => !row.data.isError))
  const efforts = log.filter(row => row.type === 'request/header').map(row => row.data.header.config.reasoningEffort)
  assert.equal(efforts.includes('max'), !faultOnly)
  if (report) assert.notEqual(efforts.at(-1), 'max')
  if (explicit) {
    assert.ok(!calls.includes('/ego/explore'), 'diagnostic motion did not reach the robot')
    assert.deepEqual(calls.filter(path => ['/ego/recovery/start', '/ego/recovery/finish', '/ego/recovery/verify',
      '/ego/recovery/verify/finish'].includes(path)), ['/ego/recovery/start', '/ego/recovery/finish',
      '/ego/recovery/verify', '/ego/recovery/verify/finish'])
  }
  if (mandatory) assert.equal(scans, 2)
  if (proactive) {
    assert.equal(picks, 2)
    assert.ok(calls.indexOf('/skill/wait') < calls.indexOf('/ego/fault/degrade'))
    assert.ok(calls.indexOf('/ego/recovery/finish') < calls.lastIndexOf('/skill/run'))
    const attempt = dispatches.find(row => row.data.name === 'ego_begin_recovery')
    assert.deepEqual(JSON.parse(attempt.data.content[0].text).diagnostic.faultJob, faultJob)
  }
  if (faultOnly) {
    assert.deepEqual(calls, ['/ego/summary', '/ego/fault/degrade', '/ego/summary', '/ego/fault/state', '/ego/recovery/task-review'])
  }
  const final = log.findLast(row => row.type === 'assistant/message')
  assert.ok(final.data.message.content.some(block => block.type === 'text'))
  assert.ok(output.includes('"type":"final"'))
})
