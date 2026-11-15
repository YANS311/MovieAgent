(function () {
  'use strict';
  const form = document.getElementById('trace-explanation-form');
  if (!form) return;
  const status = document.getElementById('trace-explanation-status');
  const output = document.getElementById('trace-explanation-output');
  let busy = false;
  form.addEventListener('submit', async event => {
    event.preventDefault(); if (busy) return; busy = true;
    const button = form.querySelector('button'); button.disabled = true;
    status.textContent = '正在核验当前账号、Trace与电影身份…'; output.textContent = '';
    try {
      const response = await fetch(form.dataset.endpoint, {method:'POST',credentials:'same-origin',
        headers:{'Content-Type':'application/json','X-CSRFToken':form.querySelector('[name=csrfmiddlewaretoken]').value},
        body:JSON.stringify({trace_id:form.dataset.traceId,movie_id:Number(form.dataset.movieId)})});
      const value = await response.json();
      if (!response.ok || value.trace_id !== form.dataset.traceId || value.movie_id !== Number(form.dataset.movieId) ||
          value.generation_source !== 'DETERMINISTIC_EVIDENCE_TEMPLATE' || value.llm_invoked !== false || typeof value.explanation !== 'string') {
        throw new Error('UNVERIFIED_EXPLANATION');
      }
      output.textContent = value.explanation;
      status.textContent = '确定性证据模板 · 未调用LLM · 原始排名与分数保持不变';
    } catch (error) { status.textContent = '解释暂不可用或授权已变化，未显示替代解释。'; }
    finally { button.disabled = false; busy = false; }
  });
}());
