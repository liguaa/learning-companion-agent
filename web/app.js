const $ = (selector) => document.querySelector(selector);
let current = null;
let sessions = [];
let busy = false;
let toastTimer;

async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '请求失败');
  return result;
}
function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}
function notify(message) {
  const toast = $('#toast'); toast.textContent = message; toast.classList.remove('hidden');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.add('hidden'), 5000);
}
function showView(name) {
  for (const view of ['chat', 'knowledge', 'settings']) {
    $(`#${view}-view`).classList.toggle('hidden', view !== name);
    $(`[data-view="${view}"]`).classList.toggle('active', view === name);
  }
  $('#sidebar').classList.remove('open');
  if (name === 'knowledge') loadDocuments();
  if (name === 'settings') loadSettings();
}
async function loadSessions() {
  sessions = await api('/api/sessions');
  const list = $('#sessions'); list.replaceChildren();
  for (const session of sessions) {
    const button = node('button', `session${current === session.id ? ' selected' : ''}`, session.title);
    button.title = session.title;
    button.onclick = () => selectSession(session.id);
    list.append(button);
  }
}
function welcome() {
  const box = $('#conversation'); box.replaceChildren();
  const wrap = node('div', 'welcome');
  wrap.append(node('div', 'welcome-icon', '✳'), node('div', 'eyebrow', 'A LITTLE PROGRESS, EVERY DAY'), node('h1', '', '今天想弄懂什么？'), node('p', '', '不必急着找到答案。带上你的问题和想法，我们一起慢慢探索。'));
  const suggestions = node('div', 'suggestions');
  for (const [icon, title, prompt] of [['◈', '理解一个新概念', '我想理解一个概念，请先问问我已经了解了什么。'], ['⌘', '一起看看代码', '请帮我分析这段代码，我会先告诉你我的思路：\n'], ['▤', '结合我的学习资料', '请结合我的知识库，引导我学习：']]) {
    const button = node('button', 'suggestion');
    button.append(node('span', 'icon', icon), node('span', '', title), node('span', 'arrow', '↗'));
    button.onclick = () => { $('#input').value = prompt; $('#input').focus(); };
    suggestions.append(button);
  }
  wrap.append(suggestions); box.append(wrap);
}
function sandboxCard(run) {
  const result = run.result;
  const labels = {success:'运行成功', error:'代码报错', timeout:'执行超时', output_limit:'输出超限', unavailable:'环境不可用', internal_error:'执行失败', invalid_input:'输入无效'};
  const card = node('section', `sandbox-card sandbox-${result.status === 'success' ? 'success' : 'failure'}`);
  const header = node('div', 'sandbox-heading');
  header.append(node('strong', '', 'Python 沙箱'), node('span', 'sandbox-status', labels[result.status] || '未知状态'));
  card.append(header);
  const details = node('details', 'sandbox-code');
  details.append(node('summary', '', '查看执行代码'));
  const code = node('pre'); code.append(node('code', '', run.code || '本次记录未保留代码。')); details.append(code);
  if (run.stdin) details.append(node('small', '', '标准输入'), node('pre', '', run.stdin));
  card.append(details);
  if (result.message) card.append(node('p', 'sandbox-description', result.message));
  if (result.stdout) card.append(node('div', 'sandbox-output-label', '标准输出'), node('pre', 'sandbox-output', result.stdout));
  else if (result.status === 'success') card.append(node('p', 'sandbox-description', '执行完成，无标准输出。'));
  if (result.stderr) card.append(node('div', 'sandbox-output-label', '错误输出'), node('pre', 'sandbox-output sandbox-stderr', result.stderr));
  const metadata = [];
  if (result.exit_code !== null && result.exit_code !== undefined) metadata.push(`退出码 ${result.exit_code}`);
  if (result.duration_seconds > 0) metadata.push(`耗时 ${Number(result.duration_seconds).toFixed(2)} 秒`);
  if (result.truncated) metadata.push('输出已截断');
  if (metadata.length) card.append(node('div', 'sandbox-meta', metadata.join(' · ')));
  if (result.cleanup_confirmed === false) card.append(node('p', 'sandbox-description', `容器清理未确认，请检查 Docker。${result.container_name || ''}`));
  if (result.status === 'unavailable') {
    const check = node('button', 'retry', '重新检查沙箱');
    check.onclick = () => checkSandbox(check, card);
    card.append(check);
  }
  return card;
}
async function checkSandbox(button, container) {
  button.disabled = true;
  const previous = button.textContent; button.textContent = '正在检查…';
  let status = container.querySelector('.sandbox-check-result');
  if (!status) { status = node('p', 'sandbox-description sandbox-check-result'); status.setAttribute('role','status'); container.append(status); }
  try { const result = await api('/api/sandbox/check', {}); status.textContent = result.message + (result.status === 'ready' ? ' 可在对话中请求重新执行；历史结果保持原样。' : ''); }
  catch (error) { status.textContent = error.message; }
  finally { button.disabled = false; button.textContent = previous; }
}
function addMessage(role, content, runs = [], renderedHtml = null) {
  const box = node('article', `message ${role}`), avatar = node('div', 'avatar', role === 'user' ? '你' : '学');
  const bubble = node('div', 'bubble');
  bubble.append(node('div', 'message-label', role === 'user' ? '你的思考' : '学习伴侣'));
  runs.forEach(run => bubble.append(sandboxCard(run)));
  const body = node('div', 'message-body');
  if (role === 'assistant' && typeof renderedHtml === 'string') {
    // Only the server's HTML-disabled Markdown renderer supplies this field.
    body.classList.add('markdown-body');
    body.innerHTML = renderedHtml;
    body.querySelectorAll('a').forEach(link => { link.target = '_blank'; link.rel = 'noopener noreferrer'; });
  } else body.textContent = content;
  bubble.append(body); box.append(avatar, bubble); $('#conversation').append(box);
}
async function selectSession(id) {
  try {
    current = id; showView('chat');
    const data = await api(`/api/sessions/${id}`);
    if (current !== id) return;
    $('#page-title').textContent = data.session.title;
    $('#conversation').replaceChildren();
    if (!data.messages.length) welcome();
    else data.messages.forEach(message => addMessage(message.role, message.content, (data.sandbox_runs || []).filter(run => run.reply_id === message.id), message.html));
    if (data.session.pending_message_id) {
      const state = node('div', 'pending', '上一次请求尚未完成。');
      const retry = node('button', 'retry', '重试请求');
      retry.onclick = () => runTurn('retry'); state.append(document.createElement('br'), retry); $('#conversation').append(state);
    }
    await loadSessions();
    $('#conversation').scrollTop = $('#conversation').scrollHeight;
  } catch (error) { notify(error.message); }
}
async function createSession() {
  const session = await api('/api/sessions', {}); await selectSession(session.id); return session.id;
}
async function runTurn(action = 'ask') {
  if (busy) return;
  const input = $('#input'); const content = input.value.trim();
  if (action === 'ask' && !content) return;
  busy = true; $('#send').disabled = true;
  try {
    if (!current) await createSession();
    const sid = current;
    if (action === 'ask') { input.value = ''; if ($('#conversation .welcome')) $('#conversation').replaceChildren(); addMessage('user', content); }
    const pending = node('div', 'pending', '正在思考，可能需要一些时间…'); $('#conversation').append(pending);
    $('#conversation').scrollTop = $('#conversation').scrollHeight;
    try { await api(`/api/sessions/${sid}/${action}`, action === 'ask' ? {content} : {}); }
    catch (error) {
      notify(error.message);
      if (action === 'ask') {
        try { const state = await api(`/api/sessions/${sid}`); if (!state.session.pending_message_id && !input.value) input.value = content; } catch (_) { /* Keep the original error visible. */ }
      }
    }
    finally { pending.remove(); if (current === sid) await selectSession(sid); else await loadSessions(); }
  } catch (error) { if (action === 'ask') input.value = content; notify(error.message); }
  finally { busy = false; $('#send').disabled = false; }
}
async function loadDocuments() {
  try {
    const documents = await api('/api/documents'), list = $('#documents'); list.replaceChildren();
    $('#doc-count').textContent = `${documents.length} 份资料`;
    if (!documents.length) list.append(node('div', 'empty', '还没有资料。导入第一份笔记，开始建立你的知识库。'));
    for (const doc of documents) {
      const row = node('div', 'doc-row'), info = node('div', 'doc-info');
      info.append(node('strong', '', doc.filename), node('small', '', `版本 ${doc.revision} · ${doc.chunks} 个片段`));
      const view = node('button', 'text-button', '查看');
      view.onclick = async () => { try { const data = await api(`/api/documents/${doc.id}`); const detail = $('#document-detail'); detail.replaceChildren(node('strong', '', data.filename), node('pre', '', data.content)); detail.classList.remove('hidden'); detail.scrollIntoView({behavior:'smooth'}); } catch (error) { notify(error.message); } };
      const remove = node('button', 'text-button danger', '删除');
      remove.onclick = async () => { if (!confirm(`删除知识库中的“${doc.filename}”？原始文件不会被删除。`)) return; try { await api(`/api/documents/${doc.id}/delete`, {}); $('#document-detail').classList.add('hidden'); await loadDocuments(); notify('资料已删除'); } catch (error) { notify(error.message); } };
      row.append(node('div', 'doc-icon', '▤'), info, view, remove); list.append(row);
    }
  } catch (error) { notify(error.message); }
}
async function loadSettings() {
  try {
    const settings = await api('/api/settings'), list = $('#settings-content'); list.replaceChildren();
    $('#model-provider').value = settings.provider;
    $('#model-name').value = settings.model;
    $('#model-url').value = settings.base_url;
    $('#model-key').value = '';
    $('#model-key').required = !settings.has_api_key;
    $('#model-key').placeholder = settings.has_api_key ? '留空保留已有密钥' : '填写 API Key';
    $('#key-status').textContent = settings.has_api_key ? '已配置' : '未配置';
    for (const [label, value] of [['数据位置', settings.database], ['代码沙箱', settings.sandbox]]) {
      const row = node('div', 'setting-row'); row.append(node('span', '', label), node('strong', '', value)); list.append(row);
    }
    const sandbox = node('div', 'sandbox-settings');
    const check = node('button', 'retry', '检查沙箱环境');
    check.onclick = () => checkSandbox(check, sandbox);
    sandbox.append(check); list.append(sandbox);
  } catch (error) { notify(error.message); }
}
$('#model-settings').onsubmit = async event => {
  event.preventDefault();
  const button = $('#save-settings'), errorBox = $('#settings-error'), result = $('#settings-result');
  button.disabled = true; button.textContent = '正在保存…';
  errorBox.classList.add('hidden'); result.textContent = '';
  try {
    await api('/api/settings', {provider: $('#model-provider').value, model: $('#model-name').value, base_url: $('#model-url').value, api_key: $('#model-key').value});
    $('#model-key').value = '';
    await loadSettings();
    result.textContent = '已保存，下一次请求生效';
  } catch (error) { errorBox.textContent = error.message; errorBox.classList.remove('hidden'); }
  finally { button.disabled = false; button.textContent = '保存配置'; }
};
$('#new-session').onclick = () => createSession().catch(error => notify(error.message));
$('#menu').onclick = () => $('#sidebar').classList.toggle('open');
document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => showView(button.dataset.view));
$('#send').onclick = () => runTurn();
$('#input').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); runTurn(); } });
$('#import-form').onsubmit = async event => { event.preventDefault(); try { const result = await api('/api/documents/import', {path: $('#import-path').value}); notify(`${result.filename}：${{imported:'已导入',updated:'已更新',unchanged:'没有变化'}[result.status]}`); $('#import-path').value = ''; await loadDocuments(); } catch (error) { notify(error.message); } };
welcome();
loadSessions().then(() => { if (sessions.length) selectSession(sessions[0].id); }).catch(error => notify(error.message));
