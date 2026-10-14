'use strict';
// Only Node builtins + the actual sendMessage source in a VM. No network/browser/Django.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

if (process.env.CUDA_VISIBLE_DEVICES !== '' ||
    ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'].some(k => process.env[k] !== '1')) {
    throw new Error('CPU-only environment required');
}
const template = fs.readFileSync(path.join(__dirname, '..', 'templates', 'agent_chat.html'), 'utf8');
const start = template.indexOf('function sendMessage() {');
const end = template.indexOf('// ═══ 快捷消息 ═══', start);
assert(start >= 0 && end > start);
const source = template.slice(start, end);
const frame = value => 'data: ' + JSON.stringify(value) + '\n\n';
const snapshot = value => value === undefined ? undefined : JSON.parse(JSON.stringify(value));
const meta = {intent: 'TEST', latency_ms: 12, recommended_ids: [9, 3],
    explanations: {'9': 'synthetic reason'}, movie_details: {'9': {title: 'synthetic movie'}},
    trace_steps: [{type: 'thought', content: 'synthetic trace'}], react_trace: {thought: 'synthetic'},
    need_clarification: true, clarification_options: ['synthetic choice']};

async function run(packets, options = {}) {
    const observed = {bubbles: [], statuses: [], sending: [], nodes: [], formatCalls: [],
        enhanced: [], react: [], cards: [], clarification: [], readCalls: 0, getReaderCalls: 0,
        bodyReads: 0, httpBodyReads: 0, fetchCalls: [], loadingRemovals: 0};
    const queue = packets.map(p => typeof p === 'string' ? new TextEncoder().encode(p) : p);
    let index = 0;
    const reader = {read() {
        observed.readCalls++;
        if (observed.readCalls > queue.length + 3) throw new Error('Unexpected repeated EOF read');
        if (options.readerErrorAt === observed.readCalls - 1) {
            return Promise.reject(new Error('synthetic-private-SQL-password-model-path'));
        }
        if (index === queue.length) return Promise.resolve({done: true});
        return Promise.resolve({done: false, value: queue[index++]});
    }};
    const response = {ok: (options.status || 200) >= 200 && (options.status || 200) < 300};
    Object.defineProperty(response, 'body', {get() {
        observed.bodyReads++;
        return options.noBody ? null : {getReader() { observed.getReaderCalls++; return reader; }};
    }});
    response.json = response.text = () => {
        observed.httpBodyReads++;
        throw new Error('Raw backend details must never be rendered');
    };
    const chat = {scrollHeight: 50, scrollTop: 0, appendChild(node) { observed.nodes.push(node); }};
    const input = {value: 'synthetic prompt'};
    const context = {
        debugMode: false, TextDecoder,
        document: {
            getElementById(id) {
                if (id === 'userInput') return input;
                if (id === 'chatWindow') return chat;
                throw new Error('Unexpected DOM lookup');
            },
            createElement() {
                const content = {innerHTML: ''};
                return {innerHTML: '', className: '', attributes: {}, content,
                    querySelector(selector) { assert.equal(selector, '.bubble-content'); return content; },
                    setAttribute(key, value) { this.attributes[key] = value; }};
            }
        },
        fetch(url, config) {
            observed.fetchCalls.push({url, config: snapshot(config)});
            if (options.fetchError) return Promise.reject(new Error('synthetic-private-URL-token'));
            return Promise.resolve(response);
        },
        appendBubble(role, text, data) { observed.bubbles.push({role, text, data: snapshot(data)}); },
        appendLoading() { return 'synthetic-loading'; },
        removeLoading() { observed.loadingRemovals++; },
        setSending(value) { observed.sending.push(value); },
        updateStatus(html) { observed.statuses.push(html); },
        formatAiText(text) { observed.formatCalls.push(text); return 'formatted:' + text; },
        appendEnhancedTrace(steps, trace) { observed.enhanced.push(snapshot({steps, trace})); },
        appendReactTrace(trace) { observed.react.push(snapshot(trace)); },
        appendRecCards(ids, explanations, details) {
            observed.cards.push(snapshot({ids, explanations, details}));
            if (options.renderError) throw new Error('synthetic-private-render-stack');
        },
        appendClarificationOptions(values) { observed.clarification.push(snapshot(values)); }
    };
    vm.createContext(context);
    vm.runInContext(source, context, {timeout: 1000});
    await context.sendMessage();  // Only the source function in a stub context, not a browser.
    assert.deepEqual(observed.sending, [true, false], 'Sending lock must always be released');
    assert.equal(observed.fetchCalls.length, 1, 'No retries or reconnects');
    const displayed = JSON.stringify({bubbles: observed.bubbles, statuses: observed.statuses});
    for (const secret of ['synthetic-private-SQL-password-model-path', 'synthetic-private-URL-token',
                          'synthetic-private-render-stack', 'raw-backend-secret']) {
        assert(!displayed.includes(secret), 'Raw internal details leaked');
    }
    return observed;
}
function aiBubbles(observed) { return observed.bubbles.filter(b => b.role === 'ai'); }
function success(observed) {
    assert.equal(observed.statuses.filter(s => s.includes('推理完成')).length, 1);
    assert(observed.statuses.at(-1).includes('推理完成'));
    assert.equal(observed.statuses.filter(s => s.includes('未成功完成')).length, 0);
}
function failure(observed) {
    assert.equal(observed.statuses.filter(s => s.includes('推理完成')).length, 0);
    assert.equal(observed.statuses.filter(s => s.includes('未成功完成')).length, 1);
    assert(observed.statuses.at(-1).includes('未成功完成'));
    assert.equal(aiBubbles(observed).length, 1, 'Exactly one failure bubble');
    assert.equal(observed.cards.length, 0);
    assert.equal(observed.enhanced.length, 0);
}

const cases = [];
const test = (name, fn) => cases.push({name, fn});

test('chunk + done + EOF preserves text, trace, cards and clarification', async () => {
    const o = await run([frame({type: 'chunk', content: 'hel'}), frame({type: 'chunk', content: 'lo'}),
                         frame({type: 'done', meta})]);
    success(o);
    assert.deepEqual(o.formatCalls, ['hel', 'hello']);
    assert.equal(o.nodes.length, 1);
    assert.equal(o.nodes[0].content.innerHTML, 'formatted:hello');
    assert.equal(o.nodes[0].attributes['data-response'], 'hello');
    assert.deepEqual(o.cards, [{ids: meta.recommended_ids, explanations: meta.explanations, details: meta.movie_details}]);
    assert.deepEqual(o.enhanced, [{steps: meta.trace_steps, trace: meta.react_trace}]);
    assert.deepEqual(o.clarification, [meta.clarification_options]);
    assert.equal(aiBubbles(o).length, 0);
});
test('error + EOF never runs success', async () => {
    failure(await run([frame({type: 'error', content: 'raw-backend-secret'})]));
});
test('chunk + persistence error + EOF keeps answer and marks failure', async () => {
    const o = await run([frame({type: 'chunk', content: 'partial answer'}),
        frame({type: 'error', code: 'AGENT_PERSISTENCE_FAILED', content: 'raw-backend-secret'})]);
    failure(o);
    assert.equal(o.nodes[0].content.innerHTML, 'formatted:partial answer');
    assert(aiBubbles(o)[0].text.includes('保存未确认完成'));
    assert.equal(o.nodes[0].attributes['data-response'], undefined);
});
test('chunk + EOF without done is incomplete', async () => {
    const o = await run([frame({type: 'chunk', content: 'answer'})]);
    failure(o);
    assert(aiBubbles(o)[0].text.includes('响应不完整'));
    assert.equal(o.nodes[0].content.innerHTML, 'formatted:answer');
});
test('empty EOF is incomplete', async () => { failure(await run([])); });
for (const status of [400, 401, 403, 500, 503]) {
    test(`HTTP ${status} never opens a stream or renders backend details`, async () => {
        const o = await run([frame({type: 'done', meta})], {status});
        failure(o);
        assert.equal(o.readCalls, 0);
        assert.equal(o.getReaderCalls, 0);
        assert.equal(o.bodyReads, 0);
        assert.equal(o.httpBodyReads, 0);
    });
}
test('reader failure before first frame is not success', async () => {
    failure(await run([], {readerErrorAt: 0}));
});
test('reader failure after chunk keeps partial answer without success', async () => {
    const o = await run([frame({type: 'chunk', content: 'partial'})], {readerErrorAt: 1});
    failure(o);
    assert.equal(o.nodes[0].content.innerHTML, 'formatted:partial');
});
test('duplicate errors only produce one failure bubble', async () => {
    failure(await run([frame({type: 'error'}), frame({type: 'error'})]));
});
test('error followed by reader failure does not duplicate failure handling', async () => {
    failure(await run([frame({type: 'error'})], {readerErrorAt: 1}));
});
test('error then done cannot become successful', async () => {
    failure(await run([frame({type: 'error'}), frame({type: 'done', meta})]));
});
test('done then error before EOF is a failure', async () => {
    failure(await run([frame({type: 'done', meta}), frame({type: 'error'})]));
});
test('done then reader exception is not transport success', async () => {
    failure(await run([frame({type: 'done', meta})], {readerErrorAt: 1}));
});
test('duplicate done does not overwrite final data or duplicate cards', async () => {
    const o = await run([frame({type: 'done', meta}), frame({type: 'done', meta: {intent: 'wrong'}})]);
    success(o);
    assert.equal(o.cards.length, 1);
    assert(o.statuses.at(-1).includes('TEST'));
    assert(!o.statuses.at(-1).includes('wrong'));
});
test('late frames after error cannot append content or overwrite failed status', async () => {
    const o = await run([frame({type: 'error'}), frame({type: 'thought', content: 'late thought'}),
                         frame({type: 'chunk', content: 'late answer'}), frame({type: 'metadata', meta}),
                         frame({type: 'done', meta})]);
    failure(o);
    assert.equal(o.nodes.length, 0);
    assert(!o.statuses.some(s => s.includes('late thought')));
});
test('metadata without done cannot render final recommendations', async () => {
    failure(await run([frame({type: 'metadata', meta})]));
});
test('metadata then done retains legacy metadata fallback', async () => {
    const o = await run([frame({type: 'metadata', meta}), frame({type: 'done'})]);
    success(o);
    assert.deepEqual(o.cards[0].ids, meta.recommended_ids);
});
test('legacy react trace fallback is unchanged on success', async () => {
    const fallback = {...meta, trace_steps: [], react_trace: {display_text: 'synthetic display'}};
    const o = await run([frame({type: 'done', meta: fallback})]);
    success(o);
    assert.deepEqual(o.react, [fallback.react_trace]);
    assert.equal(o.enhanced.length, 0);
});
test('done without chunk still renders response through existing renderer', async () => {
    const o = await run([frame({type: 'done', meta: {response: 'synthetic final response'}})]);
    success(o);
    assert.equal(aiBubbles(o)[0].text, 'synthetic final response');
});
test('fetch rejection does not expose exception or automatically retry', async () => {
    failure(await run([], {fetchError: true}));
});
test('missing readable body fails safely', async () => {
    const o = await run([], {noBody: true});
    failure(o);
    assert.equal(o.getReaderCalls, 0);
});
test('invalid JSON before done cannot be converted to success', async () => {
    failure(await run(['data: raw-backend-secret\n\n', frame({type: 'done', meta})]));
});
test('invalid event object before done is a failure', async () => {
    failure(await run([frame(null), frame({type: 'done', meta})]));
});
test('fragmented UTF-8 and event packets retain normal decoding and completion', async () => {
    const all = new TextEncoder().encode(frame({type: 'thought', content: '正在分析'}) +
        frame({type: 'chunk', content: '电影回答'}) + frame({type: 'done', meta}));
    const packets = [];
    for (let i = 0; i < all.length; i += 7) packets.push(all.slice(i, i + 7));
    const o = await run(packets);
    success(o);
    assert.equal(o.nodes[0].content.innerHTML, 'formatted:电影回答');
    assert(o.statuses.some(s => s.includes('正在分析')));
    assert.equal(o.cards.length, 1);
});
test('success renderer exception becomes a single safe failure', async () => {
    const o = await run([frame({type: 'done', meta})], {renderError: true});
    assert.equal(o.statuses.filter(s => s.includes('推理完成')).length, 0);
    assert.equal(o.statuses.filter(s => s.includes('未成功完成')).length, 1);
    assert.equal(aiBubbles(o).filter(b => b.text.includes('未成功完成')).length, 1);
});
test('terminal flags are per-request locals in the actual source', async () => {
    for (const declaration of ['var receivedDone = false;', 'var receivedError = false;', 'var terminalHandled = false;']) {
        assert.equal(source.split(declaration).length - 1, 1);
    }
    const first = await run([frame({type: 'error'})]);
    const second = await run([frame({type: 'done', meta})]);
    failure(first); success(second);
});

(async () => {
    let failures = 0;
    for (const {name, fn} of cases) {
        try { await fn(); console.log('ok - ' + name); }
        catch (error) { failures++; console.error('not ok - ' + name); console.error(error); }
    }
    console.log(`Tests: ${cases.length}, passed: ${cases.length - failures}, failed: ${failures}`);
    process.exitCode = failures ? 1 : 0;
})();
