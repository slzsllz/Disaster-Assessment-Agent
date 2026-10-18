<script setup>
import { Document, Download, Refresh, Search, Upload, Delete, Close } from '@element-plus/icons-vue'
import { ElSwitch } from 'element-plus'
import 'element-plus/es/components/switch/style/css'
import { onMounted, onUnmounted, reactive, ref } from 'vue'

defineEmits(['back'])
const documents = ref([])
const status = ref(null)
const loading = ref(false)
const uploading = ref(false)
const error = ref('')
const notice = ref('')
const fileInput = ref(null)
const busy = reactive(new Set())
const metadata = reactive({ source: '', version: '', effective_date: '', region: '', disaster_type: '' })
const query = ref('')
const searching = ref(false)
const searchResult = ref(null)
const preview = ref(null)
const statusNames = { queued: '等待处理', parsing: '解析中', embedding: '向量化中', ready: '可检索', failed: '处理失败', needs_ocr: '需要 OCR' }
let timer
let disposed = false

async function api(path, options) {
  const response = await fetch(`/api/knowledge${path}`, options)
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(typeof data.detail === 'string' ? data.detail : `请求失败（HTTP ${response.status}）`)
  }
  return response.json()
}

async function refresh(silent = false) {
  if (!silent) loading.value = true
  try {
    const [library, config] = await Promise.all([api('/documents'), api('/status')])
    if (disposed) return
    documents.value = library.documents
    status.value = config
    if (!silent) error.value = ''
  } catch (err) {
    if (!disposed) error.value = err.message
  } finally {
    loading.value = false
  }
}

async function uploadFiles(files) {
  if (uploading.value) return
  error.value = ''
  notice.value = ''
  uploading.value = true
  const failures = []
  let accepted = 0
  let duplicates = 0
  for (const file of files) {
    if (!/\.(pdf|docx)$/i.test(file.name) || file.size > 20 * 1024 * 1024 || !file.size) {
      failures.push(`${file.name}：仅支持非空的 PDF/DOCX，最大 20 MB`)
      continue
    }
    const body = new FormData()
    body.append('file', file)
    Object.entries(metadata).forEach(([key, value]) => body.append(key, value))
    try {
      const result = await api('/documents', { method: 'POST', body })
      if (result.duplicate) duplicates++
      else accepted++
    } catch (err) {
      failures.push(`${file.name}：${err.message}`)
    }
  }
  uploading.value = false
  notice.value = `已接收 ${accepted} 份文档${duplicates ? `，${duplicates} 份重复文档未重复入库` : ''}`
  await refresh(true)
  error.value = failures.join('；')
  if (fileInput.value) fileInput.value.value = ''
}

async function changeDocument(document, action, enabled) {
  if (busy.has(document.id)) return
  if (action === 'delete' && !confirm(`删除“${document.filename}”？此操作会同时删除原文和检索分块。`)) return
  busy.add(document.id)
  error.value = ''
  try {
    if (action === 'delete') {
      await api(`/documents/${document.id}`, { method: 'DELETE' })
      searchResult.value = null
    } else if (action === 'retry') {
      await api(`/documents/${document.id}/retry`, { method: 'POST' })
    } else {
      await api(`/documents/${document.id}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled }),
      })
      searchResult.value = null
    }
    await refresh(true)
  } catch (err) {
    error.value = err.message
  } finally {
    busy.delete(document.id)
  }
}

async function openPreview(document) {
  try {
    preview.value = await api(`/documents/${document.id}`)
  } catch (err) {
    error.value = err.message
  }
}

async function search() {
  if (!query.value.trim() || searching.value) return
  searching.value = true
  searchResult.value = null
  error.value = ''
  try {
    searchResult.value = await api('/search', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query.value.trim(), top_k: 5 }),
    })
  } catch (err) {
    error.value = err.message
  } finally {
    searching.value = false
  }
}

function date(value) {
  return value ? new Date(value).toLocaleString('zh-CN', { hour12: false }) : ''
}

function closeOnEscape(event) {
  if (event.key === 'Escape') preview.value = null
}

onMounted(() => {
  refresh()
  timer = setInterval(() => refresh(true), 4000)
  window.addEventListener('keydown', closeOnEscape)
})
onUnmounted(() => {
  disposed = true
  clearInterval(timer)
  window.removeEventListener('keydown', closeOnEscape)
})
</script>

<template>
  <section class="rules-page">
    <header class="rules-header">
      <div><h1>规则知识库</h1><p>项目共享 · {{ documents.length }} 份文档</p></div>
      <button class="rules-button" @click="$emit('back')">返回对话</button>
    </header>
    <div class="rules-content">
      <p v-if="error" class="rules-alert error" role="alert">{{ error }}</p>
      <p v-if="notice" class="rules-alert" role="status">{{ notice }}</p>
      <p v-if="status && !status.embedding_configured" class="rules-alert error">
        未配置 Embedding 接口。请在后端配置 RAG_EMBEDDING_BASE_URL 和 RAG_EMBEDDING_API_KEY。
      </p>

      <section class="rules-upload-section">
        <h2>添加规则文档</h2>
        <div class="rules-metadata">
          <label>发布单位<input v-model="metadata.source" maxlength="500" placeholder="如：深圳市应急管理局" /></label>
          <label>版本<input v-model="metadata.version" maxlength="100" placeholder="如：2026 年修订版" /></label>
          <label>生效日期<input v-model="metadata.effective_date" type="date" /></label>
          <label>适用地区<input v-model="metadata.region" maxlength="200" placeholder="留空为不限地区" /></label>
          <label>灾害类型<select v-model="metadata.disaster_type">
            <option value="">通用规则</option><option>洪水</option><option>地震</option>
            <option>山火</option><option>滑坡</option><option>溢油</option><option>病虫害</option>
            <option>藻华</option><option>城市综合灾害</option>
          </select></label>
        </div>
        <input ref="fileInput" class="rules-file-input" type="file" accept=".pdf,.docx" multiple @change="uploadFiles(Array.from($event.target.files || []))" />
        <button class="rules-dropzone" :disabled="uploading" @click="fileInput?.click()"
          @dragover.prevent @drop.prevent="uploadFiles(Array.from($event.dataTransfer.files))">
          <Upload /><strong>{{ uploading ? '上传中…' : '选择或拖入规则文档' }}</strong>
          <span>PDF / Word (.docx) · 单份不超过 20 MB</span>
        </button>
      </section>

      <section class="rules-list-section">
        <div class="rules-section-heading"><h2>全部文档</h2>
          <button class="rules-icon-button" title="刷新文档" :disabled="loading" @click="refresh()"><Refresh /></button>
        </div>
        <div class="rules-table-wrap">
          <table class="rules-table">
            <thead><tr><th>文档</th><th>适用范围</th><th>状态</th><th>启用</th><th>操作</th></tr></thead>
            <tbody>
              <tr v-for="document in documents" :key="document.id">
                <td><button class="rules-document-name" @click="openPreview(document)"><Document />{{ document.filename }}</button>
                  <small>{{ document.source || '未填写发布单位' }}<template v-if="document.version"> · {{ document.version }}</template></small>
                  <small v-if="document.effective_date">生效：{{ document.effective_date }}</small>
                  <small>{{ date(document.updated_at) }}</small>
                </td>
                <td>{{ document.region || '不限地区' }}<small>{{ document.disaster_type || '通用规则' }}</small></td>
                <td><span class="rules-status" :class="document.needs_reindex ? 'failed' : document.status">{{ document.needs_reindex ? '需要重建索引' : statusNames[document.status] }}</span>
                  <small v-if="document.status === 'ready'">{{ document.chunk_count }} 个分块</small>
                  <small v-if="document.error" class="rules-document-error">{{ document.error }}</small>
                </td>
                <td><el-switch :model-value="document.enabled" :disabled="busy.has(document.id)" :aria-label="`启用 ${document.filename}`" @change="changeDocument(document, 'enabled', $event)" /></td>
                <td><div class="rules-row-actions">
                  <button class="rules-icon-button" title="重试 / 重建索引" :disabled="busy.has(document.id) || ['parsing', 'embedding', 'queued'].includes(document.status)" @click="changeDocument(document, 'retry')"><Refresh /></button>
                  <a class="rules-icon-button" title="下载原文" :href="`/api/knowledge/documents/${document.id}/file`" :download="document.filename"><Download /></a>
                  <button class="rules-icon-button danger" title="删除文档" :disabled="busy.has(document.id)" @click="changeDocument(document, 'delete')"><Delete /></button>
                </div></td>
              </tr>
              <tr v-if="!documents.length"><td colspan="5" class="rules-empty">{{ loading ? '正在读取知识库…' : '暂无规则文档' }}</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="rules-search-section">
        <h2>规则检索</h2>
        <form class="rules-search-form" @submit.prevent="search">
          <input v-model="query" :disabled="searching" maxlength="2000" placeholder="输入需要查找的规则或应急处置问题" aria-label="规则检索问题" />
          <button class="rules-button primary" :disabled="searching || !query.trim()"><Search />{{ searching ? '检索中…' : '检索' }}</button>
        </form>
        <p v-if="searchResult" class="rules-search-message" :class="searchResult.status" role="status">{{ searchResult.message }}</p>
        <article v-for="result in searchResult?.results || []" :key="result.id" class="rules-search-result">
          <a :href="result.url" target="_blank" rel="noreferrer">{{ result.filename }} · {{ result.locator }}</a>
          <small>{{ result.citation }} · 向量相似度 {{ result.score.toFixed(3) }}<template v-if="result.rerank_score != null"> · 重排序得分 {{ result.rerank_score.toFixed(3) }}</template></small>
          <p>{{ result.content }}</p>
          <blockquote v-if="result.evidence_quote">核验依据：{{ result.evidence_quote }}</blockquote>
        </article>
      </section>
    </div>
    <div v-if="preview" class="rules-modal-overlay" @click.self="preview = null">
      <section class="rules-modal" role="dialog" aria-modal="true" aria-labelledby="rules-preview-title" tabindex="-1">
        <header><h2 id="rules-preview-title">{{ preview.document.filename }}</h2>
          <button class="rules-icon-button" title="关闭预览" @click="preview = null"><Close /></button>
        </header>
        <div class="rules-preview-body">
          <p v-if="!preview.preview.length">{{ preview.document.error || '文档尚未处理完成，暂无文本预览。' }}</p>
          <article v-for="(chunk, index) in preview.preview" :key="index"><strong>{{ chunk.locator }}</strong><p>{{ chunk.content }}</p></article>
        </div>
      </section>
    </div>
  </section>
</template>

<style scoped>
.rules-page { min-width: 0; height: 100%; display: flex; flex-direction: column; background: white; }
.rules-header { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 20px 32px; border-bottom: 1px solid #e5e7eb; }
.rules-header h1 { margin: 0; font-size: 22px; }
.rules-header p { margin: 6px 0 0; font-size: 13px; color: #71717a; }
.rules-content { flex: 1; min-height: 0; overflow-y: auto; padding: 8px 32px 40px; }
h2 { margin: 0 0 18px; font-size: 17px; }
.rules-content > section { padding: 26px 0; border-bottom: 1px solid #eceef1; }
.rules-metadata { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-bottom: 20px; }
.rules-metadata label { display: flex; flex-direction: column; gap: 7px; font-size: 13px; color: #52525b; }
input, select { width: 100%; min-width: 0; height: 38px; padding: 8px 10px; border: 1px solid #dcdfe6; border-radius: 6px; background: white; color: #27272a; font: inherit; }
input:focus, select:focus { outline: 2px solid #d9ecff; border-color: #409eff; }
.rules-file-input { display: none; }
.rules-dropzone { width: 100%; min-height: 122px; display: flex; align-items: center; justify-content: center; flex-direction: column; gap: 9px; border: 1px dashed #b8c6d6; border-radius: 8px; background: #fafbfd; color: #303133; }
.rules-dropzone:hover { border-color: #409eff; background: #f5f9ff; }
.rules-dropzone svg { width: 26px; height: 26px; color: #409eff; }
.rules-dropzone span { font-size: 13px; color: #71717a; }
.rules-button { display: inline-flex; align-items: center; justify-content: center; gap: 7px; min-height: 36px; padding: 8px 14px; border: 1px solid #dcdfe6; border-radius: 6px; color: #303133; background: white; white-space: nowrap; font-size: 14px; }
.rules-button.primary { background: #409eff; border-color: #409eff; color: white; }
.rules-button svg, .rules-icon-button svg, .rules-document-name svg { width: 18px; height: 18px; flex-shrink: 0; }
button:disabled { opacity: .5; cursor: not-allowed; }
.rules-section-heading { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.rules-section-heading h2 { margin: 0; }
.rules-icon-button { width: 32px; height: 32px; display: inline-flex; justify-content: center; align-items: center; border: 0; border-radius: 5px; background: transparent; color: #606266; text-decoration: none; }
.rules-icon-button:hover { color: #409eff; background: #ecf5ff; }
.rules-icon-button.danger:hover { color: #f56c6c; background: #fef0f0; }
.rules-table-wrap { overflow-x: auto; }
.rules-table { width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; }
.rules-table th { font-size: 12px; color: #71717a; background: #fafafa; font-weight: 500; }
.rules-table th, .rules-table td { padding: 14px 10px; border-bottom: 1px solid #ebeef5; vertical-align: top; }
.rules-table td:first-child { width: 38%; max-width: 360px; }
.rules-document-name { display: flex; align-items: flex-start; gap: 8px; padding: 0; background: transparent; border: 0; text-align: left; color: #303133; overflow-wrap: anywhere; font-weight: 600; }
.rules-table small, .rules-search-result small { display: block; margin-top: 6px; color: #909399; font-size: 12px; }
.rules-document-error { max-width: 260px; color: #c45656 !important; overflow-wrap: anywhere; }
.rules-row-actions { display: flex; }
.rules-status { display: inline-block; font-size: 12px; color: #606266; white-space: nowrap; }
.rules-status.ready { color: #32835a; }
.rules-status.embedding, .rules-status.parsing { color: #337ecc; }
.rules-status.failed, .rules-status.needs_ocr { color: #c45656; }
.rules-empty { text-align: center; padding: 36px !important; color: #909399; }
.rules-search-form { display: flex; gap: 10px; }
.rules-search-form input { flex: 1; font-size: 14px; }
.rules-alert { padding: 12px 14px; border-left: 3px solid #409eff; background: #ecf5ff; font-size: 14px; overflow-wrap: anywhere; }
.rules-alert.error { background: #fef0f0; border-color: #f56c6c; color: #a43a3a; }
.rules-search-message { color: #71717a; font-size: 13px; }
.rules-search-message.unavailable { color: #c45656; }
.rules-search-message.sufficient { color: #32835a; }
.rules-search-result blockquote { margin: 12px 0; padding: 8px 12px; border-left: 3px solid #c6e2ff; background: #f5f9ff; color: #606266; font-size: 13px; line-height: 1.8; white-space: pre-wrap; overflow-wrap: anywhere; }
.rules-search-result { padding: 16px 0; border-top: 1px solid #ebeef5; }
.rules-search-result a { color: #337ecc; font-size: 14px; text-decoration: none; }
.rules-search-result p, .rules-preview-body p { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 14px; line-height: 1.8; }
.rules-modal-overlay { position: fixed; inset: 0; z-index: 100; display: grid; place-items: center; padding: 20px; background: #0006; }
.rules-modal { width: min(760px, 100%); max-height: 85dvh; background: white; border-radius: 8px; display: flex; flex-direction: column; }
.rules-modal header { padding: 18px 22px; display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; border-bottom: 1px solid #ebeef5; }
.rules-modal h2 { margin: 0; overflow-wrap: anywhere; }
.rules-preview-body { overflow-y: auto; padding: 20px 24px; }
.rules-preview-body article + article { border-top: 1px solid #ebeef5; padding-top: 16px; }
@media (max-width: 1100px) { .rules-metadata { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) { .rules-header { padding: 18px 20px 18px 56px; } .rules-content { padding: 4px 16px 32px; } .rules-metadata { grid-template-columns: minmax(0, 1fr); } .rules-table { min-width: 640px; } }
</style>
