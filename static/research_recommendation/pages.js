/* Progressive page controller; scores and identity come only from the Task58 API contract. */
(function () {
  'use strict';
  const modelDatasets = {UKMAP: 'M3L_AUTHOR_ALIGNED_V1', 'PAMK-Net': 'M3L_AUTHOR_ALIGNED_V1', 'SKB-HIF': 'ML1M_CH3_V1'};
  const modes = {TEST_DOUBLE: 'TEST_DOUBLE · 页面测试数据，未执行模型推理',
    REAL_ONLINE_INFERENCE: 'REAL_ONLINE_INFERENCE · 本次真实模型评分',
    OFFLINE_MODEL_INFERENCE: 'OFFLINE_MODEL_INFERENCE · 已标记的离线模型结果',
    REALTIME_MODEL_INFERENCE: 'REALTIME_MODEL_INFERENCE · 旧接口标记，需核查执行证据'};
  function validPayload(data, model, profileId) {
    if (!data || data.model_name !== model || data.dataset !== modelDatasets[model] ||
        !Object.prototype.hasOwnProperty.call(modes, data.inference_mode) ||
        !Array.isArray(data.recommendations) || data.recommendations.length > 100 ||
        !data.profile_source || data.profile_source.mode !== 'RESEARCH_DATASET_PROFILE' ||
        data.profile_source.profile_id !== profileId ||
        data.profile_source.dataset !== data.dataset || data.profile_source.demo_only !== true ||
        !/^[a-f0-9]{64}$/.test(data.checkpoint_identity) ||
        !/^[a-f0-9]{64}$/.test(data.catalogue_manifest_identity) ||
        !/^[a-f0-9]{64}$/.test(data.profile_source.train_manifest_sha256) ||
        !/^[0-9a-f-]{36}$/.test(data.request_id)) return false;
    let previous = Infinity;
    const seen = new Set();
    return data.recommendations.every((item, index) => {
      if (!item || typeof item.canonical_movie_id !== 'string' || !/^(0|[1-9][0-9]*)$/.test(item.canonical_movie_id) || seen.has(item.canonical_movie_id) ||
          item.rank !== index + 1 || typeof item.score !== 'number' || !Number.isFinite(item.score) ||
          item.score > previous || !['VERIFIED_UNIQUE', 'AMBIGUOUS', 'UNMAPPED'].includes(item.mapping_status)) return false;
      if (item.mapping_status === 'VERIFIED_UNIQUE' &&
          (!Number.isSafeInteger(item.django_movie_id) || item.django_movie_id <= 0 || typeof item.movie_title !== 'string')) return false;
      if (item.mapping_status !== 'VERIFIED_UNIQUE' && item.django_movie_id !== null) return false;
      seen.add(item.canonical_movie_id); previous = item.score; return true;
    });
  }
  function element(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  }
  function cards(container, data, metadata, detailPrefix) {
    container.replaceChildren();
    data.recommendations.forEach(item => {
      const article = element('article', undefined, 'recommend-card');
      article.append(element('span', String(item.rank).padStart(2, '0'), 'rank-number'));
      const content = element('div');
      content.append(element('h3', item.movie_title || '未映射电影 · canonical ' + item.canonical_movie_id));
      const meta = item.mapping_status === 'VERIFIED_UNIQUE' ? metadata[String(item.django_movie_id)] : null;
      const genres = element('div', undefined, 'genre-list');
      const labels = meta && Array.isArray(meta.genres) && meta.genres.length ? meta.genres : ['类型暂缺'];
      labels.slice(0, 12).forEach(g => genres.append(element('span', String(g))));
      content.append(genres);
      content.append(element('p', '模型原始分数 ' + String(item.score), 'score'));
      content.append(element('p', 'canonical ' + item.canonical_movie_id + ' · ' + item.mapping_status, 'mapping-status'));
      if (item.mapping_status === 'VERIFIED_UNIQUE') {
        const link = element('a', '查看电影详情 ↗', 'detail-link');
        link.href = detailPrefix + item.django_movie_id + '/?trace_id=' + encodeURIComponent(data.request_id); content.append(link);
      } else {
        content.append(element('p', item.mapping_status === 'AMBIGUOUS' ? '映射存在歧义，未关联电影详情。' : '暂无可信电影映射，保留原始标识。', 'mapping-status'));
      }
      if (meta && typeof meta.poster_url === 'string' && /^(https?:\/\/|\/(static|media)\/)/.test(meta.poster_url)) {
        const image = element('img'); image.src = meta.poster_url; image.alt = item.movie_title + '海报';
        image.className = 'mini-poster'; image.loading = 'lazy'; image.referrerPolicy = 'no-referrer'; posterFallback(image); article.append(image);
      }
      article.append(content); container.append(article);
    });
  }
  // No substitute artwork: inaccessible posters get an explicit placeholder.
  function posterFallback(image) {
    const show = () => {
      if (!image.parentNode) return;
      const placeholder = element('div', undefined, 'poster-empty');
      if (image.classList.contains('mini-poster')) placeholder.classList.add('mini-poster');
      placeholder.append(element('span', '▤', 'film-symbol'), element('span', '海报暂不可用'));
      image.replaceWith(placeholder);
    };
    image.addEventListener('error', show, {once: true});
    if (image.complete && image.currentSrc && image.naturalWidth === 0) show();
  }
  document.querySelectorAll('.poster-wrap img, .detail-poster img').forEach(posterFallback);
  const form = document.getElementById('recommend-form');
  if (!form) return;
  const profile = document.getElementById('research-profile');
  const model = document.getElementById('model-name');
  const topK = document.getElementById('top-k');
  const status = document.getElementById('recommend-status');
  const results = document.getElementById('recommend-results');
  const provenance = document.getElementById('provenance-panel');
  let busy = false;
  function state(message, error, loading) {
    status.textContent = message;
    status.className = 'state-panel' + (error ? ' is-error' : '') + (loading ? ' is-loading' : '');
    status.setAttribute('aria-busy', loading ? 'true' : 'false');
  }
  function alignModel() {
    const dataset = profile.selectedOptions[0].dataset.dataset;
    document.getElementById('selected-profile-summary').textContent = dataset + ' · canonical ' + profile.selectedOptions[0].dataset.canonical;
    Array.from(model.options).forEach(option => { option.disabled = option.dataset.dataset !== dataset; });
    if (model.selectedOptions[0].disabled) model.value = Array.from(model.options).find(option => !option.disabled).value;
  }
  profile.addEventListener('change', alignModel); alignModel();
  form.addEventListener('submit', async event => {
    event.preventDefault(); if (busy || !form.reportValidity()) return;
    const k = Number(topK.value);
    if (!Number.isInteger(k) || k < 1 || k > 100) { state('Top-K应为1到100之间的整数。', true); return; }
    const request = {model_name: model.value, profile_id: profile.value, top_k: k};
    const controls = Array.from(form.querySelectorAll('input,select,button'));
    const token = form.querySelector('[name=csrfmiddlewaretoken]').value;
    busy = true; controls.forEach(el => { el.disabled = true; });
    results.replaceChildren(); provenance.hidden = true;
    state('正在请求推荐，请稍候…', false, true);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(form.dataset.endpoint, {method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type': 'application/json', 'X-CSRFToken': token}, body: JSON.stringify(request), signal: controller.signal});
      const data = await response.json();
      if (!response.ok) {
        const messages = {401: '请先登录后再生成推荐。', 403: '权限不足或会话校验失败（403），请刷新页面或重新登录。',
          404: '当前账号没有该研究画像的权限。', 409: '画像与模型的数据集不匹配，或映射版本不可用。',
          503: '模型服务暂不可用（503）。未生成替代推荐。'};
        state(messages[response.status] || '推荐请求未完成，请检查输入或稍后重试。', true); return;
      }
      if (!validPayload(data, request.model_name, request.profile_id) || data.recommendations.length > k) throw new Error('INVALID_CONTRACT');
      let metadata = {};
      const ids = data.recommendations.filter(x => x.mapping_status === 'VERIFIED_UNIQUE').map(x => x.django_movie_id);
      if (ids.length) {
        try {
          const extra = await fetch(form.dataset.metadata + '?ids=' + ids.join(','), {credentials: 'same-origin', signal: controller.signal});
          if (extra.ok) metadata = (await extra.json()).movies || {};
        } catch (error) { /* optional metadata cannot replace scores or IDs */ }
      }
      document.getElementById('inference-mode').textContent = modes[data.inference_mode];
      const list = document.getElementById('provenance-list'); list.replaceChildren();
      [['模型 / 数据集', data.model_name + ' / ' + data.dataset], ['画像身份', 'RESEARCH_DATASET_PROFILE · demo_only'],
        ['Checkpoint', data.checkpoint_identity], ['映射版本', data.mapping_version],
        ['候选范围', data.candidate_scope], ['来源', data.source_reference], ['请求 / Trace ID', data.request_id]].forEach(([label, value]) => {
        list.append(element('dt', label), element('dd', String(value)));
      });
      provenance.hidden = false;
      const traceLink = element('a', '查看本次Trace JSON ↗', 'detail-link');
      traceLink.href = form.dataset.traceTemplate.replace('00000000-0000-0000-0000-000000000000', data.request_id);
      list.append(element('dt', '可追溯记录'), element('dd'));
      list.lastElementChild.append(traceLink);
      cards(results, data, metadata, form.dataset.detailPrefix);
      state(data.recommendations.length ? '已返回' + data.recommendations.length + '部电影，按模型原始分数排序。' :
        '本次返回空结果，没有可展示的推荐电影。系统未补造候选。', false);
    } catch (error) {
      state(error.name === 'AbortError' ? '请求已超时。服务端可能仍在处理，请勿连续提交。' :
        '推荐响应异常，未展示不可信结果。请稍后重试。', true);
    } finally {
      clearTimeout(timer); busy = false; controls.forEach(el => { el.disabled = false; }); alignModel();
    }
  });
}());
