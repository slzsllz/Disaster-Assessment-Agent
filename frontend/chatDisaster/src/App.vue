<script setup>
import '@arcgis/core/assets/esri/themes/light/main.css'
import esriConfig from '@arcgis/core/config'
import Graphic from '@arcgis/core/Graphic'
import GraphicsLayer from '@arcgis/core/layers/GraphicsLayer'
import Map from '@arcgis/core/Map'
import MapView from '@arcgis/core/views/MapView'
import { Bottom, Document, Edit, Plus, Top } from '@element-plus/icons-vue'
import { ElSlider } from 'element-plus'
import 'element-plus/es/components/slider/style/css'
import { computed, nextTick, onMounted, ref } from 'vue'
import watermarkUrl from './assets/team.png'
import logoUrl from './assets/szu-logo.png'

const systemPrompt = ref(
  `You are Disaster Detection Agent, a disaster detection and remote-sensing intelligent assessment assistant.

Your role is to help users analyze remote-sensing images and geospatial data for disaster detection, disaster impact assessment, and environmental risk interpretation. You can use available tools to perform specialized tasks such as earthquake building damage assessment, flood inundation extraction, wildfire burned-area change detection, landslide segmentation, oil spill detection, crop pest-affected area detection, algal bloom candidate detection, geographic context and nearby facility query, remote-sensing index calculation, geospatial statistics, and general GeoAI analysis.

When answering, follow these rules:

1. Understand the user's intent first.
   - Determine which disaster or environmental phenomenon the user is asking about.
   - Determine whether the task requires tool execution, image interpretation, file analysis, or only an explanatory answer.
   - If the user uploads files, infer their roles from filename, content, metadata, and context when possible.

2. Use tools when needed.
   - Select the most appropriate tool according to the task type and input data.
   - For pre/post disaster tasks, identify pre-disaster and post-disaster images carefully before calling the tool.
   - When a tool returns 'Result saved at /path/to/file', you MUST use the full returned path in any subsequent tool calls.
   - Do not invent results. Base quantitative conclusions on tool outputs, generated summaries, readable metadata, or attached image content.

3. Interpret results like a remote-sensing disaster analyst.
   - Summarize the key detection results clearly.
   - Explain what the detected areas mean in practical disaster-assessment terms.
   - Include important numbers such as area, pixel count, percentage, damage level, confidence, or class distribution when available.
   - Mention limitations when relevant, such as model uncertainty, image quality, missing bands, cloud cover, lack of ground truth, binary-mask limitations, or index-threshold uncertainty.

4. Answer style.
   - Use Chinese by default unless the user asks otherwise.
   - Keep the answer clear, professional, and easy to read.
   - Do not expose local absolute file paths in the final natural-language answer.
   - The first-pass answer should focus only on the main disaster analysis and conclusion.
   - Do not include "文件说明", "输出文件", "可下载文件", or similar file-list sections in the first-pass answer.
   - Do not force a "下一步建议" section in the first-pass answer; follow-up suggestions will be added by the second-pass multimodal review when useful.

Finish your final response with a clearly labelled conclusion block:
Put the complete user-facing main analysis inside <Conclusion>. Do not write
user-facing analysis outside the <Conclusion> block.

<Conclusion>
你的灾害检测主体分析和结论
</Conclusion>`
)
const recursionLimit = ref(40)
const maxExecutionTime = ref(600)
const showTrace = ref(false)
const inputText = ref('')
const attachments = ref([])
const fileInputRef = ref(null)
const sidebarOpen = ref(true)
const mapDrawerOpen = ref(false)
const isSending = ref(false)
const conversationHistory = ref([])
const sessionId = ref(readSessionIdFromUrl() || crypto.randomUUID())
const messages = ref([])
const chatContentRef = ref(null)
const showScrollBottom = ref(false)
const arcgisContainerRef = ref(null)
const arcgisView = ref(null)
const arcgisGraphicsLayer = ref(null)
const mapLoading = ref(false)
const mapError = ref('')
const mapViewMode = ref('standard')
const activeMapAssessment = ref(null)
const mapNotice = ref('暂无空间范围')
const arcgisKey = import.meta.env.VITE_ARCGIS_API_KEY || ''

// PDF 报告预览
const pdfPreviewVisible = ref(false)
const pdfPreviewUrl = ref('')
const pdfPreviewName = ref('')

function openPdfPreview(report) {
  pdfPreviewUrl.value = report.url
  pdfPreviewName.value = report.name || 'report.pdf'
  pdfPreviewVisible.value = true
}

function closePdfPreview() {
  pdfPreviewVisible.value = false
}

const hasMessages = computed(() => messages.value.length > 0)

function readSessionIdFromUrl() {
  try {
    return new URL(window.location.href).searchParams.get('session_id') || ''
  } catch {
    return ''
  }
}

function writeSessionIdToUrl(id) {
  try {
    const url = new URL(window.location.href)
    if (id) {
      url.searchParams.set('session_id', id)
    } else {
      url.searchParams.delete('session_id')
    }
    window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`)
  } catch {
    // URL state is a convenience for refresh recovery; ignore unsupported environments.
  }
}

// 按更新时间排序（从新到旧）
const sortedConversationHistory = computed(() => {
  return [...conversationHistory.value].sort((a, b) => {
    const timeA = a.updatedAt || 0
    const timeB = b.updatedAt || 0
    return timeB - timeA // 降序：最新的在前
  })
})

const currentConversation = computed(() => {
  return conversationHistory.value.find((item) => item.id === sessionId.value) || null
})

function compactTitle(value) {
  const title = String(value || '').replace(/\s+/g, ' ').trim()
  if (!title) return ''
  return title.length > 32 ? `${title.slice(0, 32)}...` : title
}

const currentConversationTitle = computed(() => {
  const savedTitle = currentConversation.value?.title
  if (savedTitle && savedTitle !== '新对话') return savedTitle
  const firstUserMessage = messages.value.find((item) => item.role === 'user')?.content
  return compactTitle(firstUserMessage) || '新对话'
})

function formatDateTime(timestamp) {
  if (!timestamp) return '暂无更新时间'
  const date = new Date(timestamp)
  const pad = (value) => String(value).padStart(2, '0')
  const monthDayTime = `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`
  if (date.getFullYear() === new Date().getFullYear()) {
    return monthDayTime
  }
  return `${date.getFullYear()}-${monthDayTime}`
}

const currentConversationUpdatedAt = computed(() => {
  const savedTime = currentConversation.value?.updatedAt || 0
  if (savedTime) return formatDateTime(savedTime)
  if (messages.value.length) return formatDateTime(Date.now())
  return '尚未开始对话'
})
const previewableImageTypes = new Set([
  'image/png',
  'image/jpeg',
  'image/gif',
  'image/webp',
  'image/svg+xml',
])

function formatSize(bytes) {
  if (!bytes) return '0 KB'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = bytes
  let index = 0
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024
    index += 1
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
}

function isTableRow(line) {
  return /^\|.*\|$/.test(line.trim())
}

function isTableDivider(line) {
  return /^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line.trim())
}

function parseTableCells(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim())
}

function renderTable(rows) {
  if (rows.length < 2 || !isTableDivider(rows[1])) return ''
  const headers = parseTableCells(rows[0])
  const bodyRows = rows.slice(2).filter(isTableRow).map(parseTableCells)
  const head = headers.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join('')
  const body = bodyRows
    .map((cells) => `<tr>${cells.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join('')}</tr>`)
    .join('')
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`
}

function stripArtifactBlock(value) {
  return String(value ?? '')
    .replace(/<Artifacts>[\s\S]*?<\/Artifacts>/gi, '')
    .replace(/<Artifacts>[\s\S]*$/i, '')
}

function renderMarkdown(value) {
  const lines = stripArtifactBlock(value).split('\n')
  const html = []
  let listOpen = false

  function closeList() {
    if (listOpen) {
      html.push('</ul>')
      listOpen = false
    }
  }

  for (let index = 0; index < lines.length; index += 1) {
    const rawLine = lines[index]
    const line = rawLine.trim()
    if (!line) {
      closeList()
      continue
    }

    if (isTableRow(line) && isTableDivider(lines[index + 1] || '')) {
      closeList()
      const tableRows = [line, lines[index + 1].trim()]
      index += 2
      while (index < lines.length && isTableRow(lines[index])) {
        tableRows.push(lines[index].trim())
        index += 1
      }
      index -= 1
      html.push(renderTable(tableRows))
      continue
    }

    const heading = /^(#{1,4})\s+(.+)$/.exec(line)
    if (heading) {
      closeList()
      const level = Math.min(heading[1].length + 1, 4)
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`)
      continue
    }

    const ordered = /^\d+\.\s+(.+)$/.exec(line)
    const unordered = /^[-*]\s+(.+)$/.exec(line)
    if (ordered || unordered) {
      if (!listOpen) {
        html.push('<ul>')
        listOpen = true
      }
      html.push(`<li>${renderInlineMarkdown((ordered || unordered)[1])}</li>`)
      continue
    }

    closeList()
    html.push(`<p>${renderInlineMarkdown(line)}</p>`)
  }

  closeList()
  return html.join('')
}

function outputFileLabel(name = '') {
  const lower = String(name).toLowerCase()
  if (lower.includes('damage_overlay')) return '建筑损坏叠加可视化图'
  if (lower.includes('building_mask')) return '建筑物检测掩膜图'
  if (lower.includes('damage_mask')) return '建筑损坏等级掩膜图'
  if (lower.includes('flood_overlay')) return '洪水淹没叠加可视化图'
  if (lower.includes('flood_mask') && lower.endsWith('.tif')) return '洪水淹没掩膜 GeoTIFF'
  if (lower.includes('flood_mask')) return '洪水淹没掩膜图片'
  if (lower.includes('burned_area_overlay')) return '山火烧毁区叠加可视化图'
  if (lower.includes('burned_area_comparison')) return '山火变化检测对比图'
  if (lower.includes('burned_area_mask')) return '山火烧毁区掩膜图片'
  if (lower.includes('burned_area_metrics')) return '山火烧毁区指标表 CSV'
  if (lower.includes('landslide_vis')) return '滑坡识别叠加可视化图'
  if (lower.includes('landslide_mask')) return '滑坡区域掩膜图片'
  if (lower.includes('oil_vis')) return '海面溢油叠加可视化图'
  if (lower.includes('oil_mask')) return '海面溢油掩膜图片'
  if (lower.includes('pest_vis')) return '受害植株/区域检测框图'
  if (lower.includes('true_color_rgb')) return 'Sentinel-2 真彩色预览图'
  if (lower.includes('mndwi_water_mask')) return 'MNDWI 水体掩膜图'
  if (lower.includes('mndwi_heatmap')) return 'MNDWI 水体指数热力图'
  if (lower.includes('water_ndci_heatmap')) return '水体区域 NDCI 热力图'
  if (lower.includes('ndci_bloom_overlay')) return '候选藻华叠加可视化图'
  if (lower.includes('ndci_histogram')) return '水体 NDCI 直方图'
  if (lower.includes('ndci_comparison')) return '藻华检测综合对比图'
  if (lower.includes('ndci_bloom_mask')) return '候选藻华掩膜 GeoTIFF'
  if (lower === 'stats.json') return 'NDCI 统计诊断 JSON'
  if (lower === 'summary.json' || lower.endsWith('_summary.json')) return '摘要报告 JSON'
  if (lower.endsWith('.geojson')) return '矢量结果 GeoJSON'
  if (lower.endsWith('.tif') || lower.endsWith('.tiff')) return 'GeoTIFF 栅格结果'
  if (lower.endsWith('.csv')) return '结果表 CSV'
  if (lower.endsWith('.json')) return '摘要报告 JSON'
  if (lower.endsWith('.npy')) return 'NumPy 数据文件'
  return name || '输出文件'
}

function resultImageCaption(image) {
  return outputFileLabel(image?.name || '')
}

function imageLegendItems(image, message) {
  if (!message.legend?.length) return []
  const lower = String(image?.name || '').toLowerCase()
  const canExplainLegend = [
    'overlay',
    'comparison',
    'vis',
  ].some((marker) => lower.includes(marker))
  return canExplainLegend ? message.legend : []
}

// ---------------------------------------------------------------------------
// 后端数据加载 -- 会话历史与消息从数据库读取 (DB -> API -> 前端)
// 失败时不使用浏览器本地缓存，历史对话只以数据库为准。
// ---------------------------------------------------------------------------
function mapApiMessage(m) {
  const metaParts = []
  if (m.elapsed_seconds != null) metaParts.push(`${Number(m.elapsed_seconds).toFixed(1)}s`)
  if (m.tool_call_count != null) metaParts.push(`${m.tool_call_count} tool call(s)`)
  return {
    id: String(m.id),
    role: m.role,
    content: m.content || '',
    meta: metaParts.join(' · '),
    error: '',
    images: (m.images || []).map((img) => ({ name: img.name || 'image', url: img.url })),
    legend: m.legend || [],
    report: m.report || null,
    attachments: (m.attachments || []).map((a) => {
      const name = a.name || 'file'
      const url = a.url || ''
      const ext = name.split('.').pop().toLowerCase()
      const isImage = a.type
        ? previewableImageTypes.has(a.type)
        : ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff'].includes(ext)
      // 只有用户上传的图片才显示预览；模型输出文件保持下载按钮
      const preview = m.role === 'user' && isImage ? url : ''
      return {
        id: url || name || a.path || 'file',
        name,
        size: a.size || 0,
        type: a.type || '',
        url,
        preview,
      }
    }),
  }
}

function mapApiSession(s) {
  const title = s.title || (s.first_message ? s.first_message.replace(/\s+/g, ' ').trim() : '')
  return {
    id: s.id,
    title: title.length > 24 ? `${title.slice(0, 24)}...` : title || '新对话',
    updatedAt: s.updated_at ? new Date(s.updated_at).getTime() : 0,
    messages: [], // lazy-loaded on open
  }
}

async function fetchSessions() {
  try {
    const res = await fetch('/api/sessions')
    if (!res.ok) return
    const data = await res.json()
    conversationHistory.value = (data.sessions || []).map(mapApiSession).slice(0, 30)
  } catch {
    conversationHistory.value = []
  }
}

async function fetchMessages(id) {
  try {
    const res = await fetch(`/api/sessions/${id}/messages`)
    if (!res.ok) return null
    const data = await res.json()
    return (data.messages || []).map(mapApiMessage)
  } catch {
    return null
  }
}

function releasePendingFiles() {
  attachments.value.forEach((item) => {
    if (item.preview) URL.revokeObjectURL(item.preview)
  })
  attachments.value = []
  resetFileInput()
}

function resetFileInput() {
  if (fileInputRef.value) {
    fileInputRef.value.value = ''
  }
}

function startNewConversation() {
  releasePendingFiles()
  messages.value = []
  showScrollBottom.value = false
  sessionId.value = crypto.randomUUID()
  writeSessionIdToUrl('')
  clearMapGeometry()
}

async function openConversation(conversation) {
  if (conversation.id === sessionId.value) return
  releasePendingFiles()
  sessionId.value = conversation.id
  writeSessionIdToUrl(conversation.id)
  messages.value = []
  const fetched = await fetchMessages(conversation.id)
  if (fetched && fetched.length) {
    messages.value = fetched
  }
  const hasGeometry = await showLatestSessionGeometry()
  if (!hasGeometry) clearMapGeometry()
  showScrollBottom.value = false
  scrollToBottom()
}

async function deleteConversation(conversation) {
  if (!confirm(`确定要删除对话"${conversation.title}"吗？`)) return

  try {
    const res = await fetch(`/api/sessions/${conversation.id}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
  } catch (err) {
    console.error('删除对话失败:', err)
    alert('删除失败，请重试')
    return
  }

  // 如果删除的是当前对话，开启新对话
  if (conversation.id === sessionId.value) {
    startNewConversation()
  }
  await fetchSessions()
}

const ARCGIS_BASEMAPS = {
  standard: 'arcgis/navigation',
  satellite: 'arcgis/imagery',
}

function applyArcgisBasemap() {
  if (!arcgisView.value?.map) return
  arcgisView.value.map.basemap = ARCGIS_BASEMAPS[mapViewMode.value] || ARCGIS_BASEMAPS.standard
}

function setMapViewMode(mode) {
  mapViewMode.value = mode
  applyArcgisBasemap()
}

function clearMapGeometry() {
  arcgisGraphicsLayer.value?.removeAll()
  activeMapAssessment.value = null
  mapNotice.value = '暂无空间范围'
}

function geometryToRings(geom) {
  if (!geom) return []
  const geometry = typeof geom === 'string' ? JSON.parse(geom) : geom
  if (geometry.type === 'Polygon') {
    return [geometry.coordinates?.[0] || []]
  }
  if (geometry.type === 'MultiPolygon') {
    return (geometry.coordinates || []).map((polygon) => polygon?.[0] || []).filter(Boolean)
  }
  return []
}

async function drawAssessmentGeometry(assessment) {
  if (!assessment?.geom) return false
  await openMapDrawer()
  if (!arcgisView.value || !arcgisGraphicsLayer.value) return false

  let rings = []
  try {
    rings = geometryToRings(assessment.geom)
  } catch {
    mapNotice.value = '空间范围解析失败'
    return false
  }
  if (!rings.length) return false

  arcgisGraphicsLayer.value.removeAll()

  const graphics = rings
    .map((ring) => ring
      .filter((point) => Array.isArray(point) && point.length >= 2)
      .map(([lng, lat]) => [Number(lng), Number(lat)]))
    .filter((ring) => ring.length >= 3)
    .map((ring) => new Graphic({
      geometry: {
        type: 'polygon',
        rings: [ring],
        spatialReference: { wkid: 4326 },
      },
      symbol: {
        type: 'simple-fill',
        color: [239, 68, 68, 0.18],
        outline: {
          color: [239, 68, 68, 0.95],
          width: 2,
        },
      },
    }))

  if (!graphics.length) return false
  arcgisGraphicsLayer.value.addMany(graphics)
  await arcgisView.value.goTo(graphics, {
    duration: 600,
    padding: { top: 48, right: 48, bottom: 48, left: 48 },
  }).catch(() => {})
  activeMapAssessment.value = assessment
  mapNotice.value = `${assessment.task || '分析结果'} 空间范围`
  return true
}

async function showLatestSessionGeometry() {
  try {
    const res = await fetch(`/api/sessions/${sessionId.value}/latest-geometry`)
    if (!res.ok) return false
    const data = await res.json()
    if (!data.found || !data.assessment?.geom) return false
    return drawAssessmentGeometry(data.assessment)
  } catch {
    return false
  }
}

async function initArcgisMap() {
  if (arcgisView.value || mapLoading.value) return

  mapLoading.value = true
  mapError.value = ''
  try {
    if (!arcgisKey) {
      throw new Error('缺少 VITE_ARCGIS_API_KEY，请在前端环境变量中配置 ArcGIS API Key。')
    }
    esriConfig.apiKey = arcgisKey
    await nextTick()
    if (!arcgisContainerRef.value) return

    arcgisGraphicsLayer.value = new GraphicsLayer()
    const map = new Map({
      basemap: ARCGIS_BASEMAPS[mapViewMode.value],
      layers: [arcgisGraphicsLayer.value],
    })
    arcgisView.value = new MapView({
      container: arcgisContainerRef.value,
      map,
      center: [104.1954, 35.8617],
      zoom: 3,
    })
  } catch (error) {
    mapError.value = error.message || String(error)
  } finally {
    mapLoading.value = false
  }
}

async function openMapDrawer() {
  mapDrawerOpen.value = true
  await nextTick()
  await initArcgisMap()
  nextTick(() => {
    arcgisView.value?.resize()
  })
}

function handleFiles(event) {
  const files = Array.from(event.target.files || [])
  attachments.value = files.map((file) => ({
    id: `${file.name}-${file.lastModified}`,
    file,
    name: file.name,
    size: file.size,
    type: file.type,
    preview: previewableImageTypes.has(file.type) ? URL.createObjectURL(file) : '',
  }))
  resetFileInput()
}

function clearAttachment(id) {
  const item = attachments.value.find((file) => file.id === id)
  if (item?.preview) URL.revokeObjectURL(item.preview)
  attachments.value = attachments.value.filter((file) => file.id !== id)
}

function fileExtension(name) {
  const parts = String(name || '').split('.')
  return parts.length > 1 ? parts.pop().toUpperCase() : 'FILE'
}

function isChatAtBottom() {
  const el = chatContentRef.value
  if (!el) return true
  return el.scrollHeight - el.scrollTop - el.clientHeight < 80
}

function updateScrollBottomButton() {
  showScrollBottom.value = hasMessages.value && !isChatAtBottom()
}

function scrollToBottom(behavior = 'auto') {
  nextTick(() => {
    const el = chatContentRef.value
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior })
    showScrollBottom.value = false
  })
}

function followBottomIfNeeded() {
  const shouldFollow = !showScrollBottom.value
  nextTick(() => {
    if (shouldFollow) {
      scrollToBottom()
    } else {
      updateScrollBottomButton()
    }
  })
}

function createChatFormData(text, localAttachments) {
  const formData = new FormData()
  formData.append('session_id', sessionId.value)
  formData.append('message', text)
  formData.append('system_prompt', systemPrompt.value)
  formData.append('recursion_limit', String(recursionLimit.value))
  formData.append('max_execution_time', String(maxExecutionTime.value))
  formData.append('show_trace', String(showTrace.value))
  localAttachments.forEach((item) => {
    formData.append('files', item.file, item.name)
  })
  return formData
}

function handleStreamBlock(block, assistantId) {
  const lines = block.split('\n')
  let eventName = 'message'
  const dataLines = []

  for (const line of lines) {
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart())
    }
  }

  if (!dataLines.length) return

  const payload = JSON.parse(dataLines.join('\n'))
  const message = messages.value.find((item) => item.id === assistantId)
  if (!message) return

  if (eventName === 'delta') {
    message.content += payload.text || ''
  } else if (eventName === 'status') {
    message.meta = payload.message || ''
  } else if (eventName === 'done') {
    message.content = payload.answer || message.content || '(empty response)'
    message.meta = `${Number(payload.elapsed || 0).toFixed(1)}s · ${payload.tool_calls || 0} tool call(s)`
    message.images = payload.images || []
    message.legend = payload.legend || []
    message.attachments = (payload.files || []).map((file) => ({
      id: file.url || file.name,
      name: file.name || 'file',
      url: file.url || '',
      preview: '',
    }))
    if (payload.geometry) {
      drawAssessmentGeometry({ task: '当前分析结果', geom: payload.geometry })
    }
    message.error = ''
  } else if (eventName === 'report') {
    message.report = payload
  } else if (eventName === 'error') {
    message.content = payload.answer || '后端调用失败'
    message.meta = ''
    message.images = []
    message.legend = []
    message.attachments = []
    message.error = payload.error || ''
  }
  followBottomIfNeeded()
}

async function sendMessage() {
  if (isSending.value) return
  const text = inputText.value.trim()
  if (!text && attachments.value.length === 0) return

  const localAttachments = [...attachments.value]
  writeSessionIdToUrl(sessionId.value)
  messages.value.push({
    id: crypto.randomUUID(),
    role: 'user',
    content: text || 'Please analyze the uploaded file(s).',
    attachments: localAttachments,
  })

  const assistantId = crypto.randomUUID()
  messages.value.push({
    id: assistantId,
    role: 'assistant',
    content: '',
    meta: '正在思考...',
  })
  scrollToBottom()

  inputText.value = ''
  attachments.value = []
  isSending.value = true

  // 如果是新会话，先临时加入侧边栏历史列表，避免生成过程中看不到当前会话
  const existing = conversationHistory.value.find((item) => item.id === sessionId.value)
  if (!existing) {
    conversationHistory.value.unshift({
      id: sessionId.value,
      title: compactTitle(text) || '新对话',
      updatedAt: Date.now(),
      messages: [],
    })
  }

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      body: createChatFormData(text, localAttachments),
    })
    if (response.status === 404) {
      const fallbackResponse = await fetch('/api/chat', {
        method: 'POST',
        body: createChatFormData(text, localAttachments),
      })
      if (!fallbackResponse.ok) {
        throw new Error(`HTTP ${fallbackResponse.status}`)
      }
      const data = await fallbackResponse.json()
      const message = messages.value.find((item) => item.id === assistantId)
      if (message) {
        message.content = data.answer || '(empty response)'
        message.meta = `${Number(data.elapsed || 0).toFixed(1)}s · ${data.tool_calls || 0} tool call(s)`
        message.images = data.images || []
        message.legend = data.legend || []
        message.attachments = (data.files || []).map((file) => ({
          id: file.url || file.name,
          name: file.name || 'file',
          url: file.url || '',
          preview: '',
        }))
        if (data.geometry) {
          await drawAssessmentGeometry({ task: '当前分析结果', geom: data.geometry })
        }
        message.report = data.report || null
        message.error = data.error || ''
      }
      scrollToBottom()
      return
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    if (!response.body) {
      throw new Error('ReadableStream is not available')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const blocks = buffer.split('\n\n')
      buffer = blocks.pop() || ''
      blocks.forEach((block) => handleStreamBlock(block, assistantId))
    }

    buffer += decoder.decode()
    if (buffer.trim()) {
      handleStreamBlock(buffer, assistantId)
    }
  } catch (error) {
    const message = messages.value.find((item) => item.id === assistantId)
    if (message) {
      message.content = `后端连接失败：${error.message}`
      message.meta = ''
      message.error = String(error)
    }
    scrollToBottom()
  } finally {
    isSending.value = false
    await fetchSessions()
    await showLatestSessionGeometry()
    // AI 总结标题在后台线程执行，稍后再刷新一次以显示 AI 生成的 title
    setTimeout(() => fetchSessions(), 3500)
  }
}

onMounted(async () => {
  // 用数据库会话刷新历史侧栏。
  await fetchSessions()
  const urlSessionId = readSessionIdFromUrl()
  if (!urlSessionId) return

  sessionId.value = urlSessionId
  const fetched = await fetchMessages(urlSessionId)
  if (fetched) {
    messages.value = fetched
  }
  const hasGeometry = await showLatestSessionGeometry()
  if (!hasGeometry) clearMapGeometry()
  showScrollBottom.value = false
  scrollToBottom()
})

</script>

<template>
  <main class="app-shell" :class="{ 'sidebar-closed': !sidebarOpen }">
    <button
      class="drawer-toggle"
      type="button"
      :aria-label="sidebarOpen ? 'Hide sidebar' : 'Show sidebar'"
      @click="sidebarOpen = !sidebarOpen"
    >
      {{ sidebarOpen ? '‹' : '☰' }}
    </button>

    <aside class="sidebar">
      <img class="sidebar-logo" :src="logoUrl" alt="深圳大学" />
      <div class="sidebar-title">灾害检测助手</div>

      <section class="sidebar-section">
        <h2>高级设置</h2>
        <label class="slider-label">
          <span>智能体最大执行步数</span>
          <strong>{{ recursionLimit }}</strong>
        </label>
        <el-slider
          v-model="recursionLimit"
          :min="10"
          :max="100"
          :step="1"
          size="small"
        />

        <label class="slider-label">
          <span>智能体最长执行时间（秒）</span>
          <strong>{{ maxExecutionTime }}</strong>
        </label>
        <el-slider
          v-model="maxExecutionTime"
          :min="60"
          :max="1800"
          :step="1"
          size="small"
        />

        <label class="checkbox-row">
          <input v-model="showTrace" type="checkbox" />
          <span>显示工具调用轨迹</span>
        </label>
      </section>

      <button class="new-chat-button" type="button" @click="startNewConversation">
        <Edit />
        <span>创建新对话</span>
      </button>

      <section class="history-section">
        <h2>历史对话</h2>
        <div v-if="conversationHistory.length" class="history-list">
          <div
            v-for="conversation in sortedConversationHistory"
            :key="conversation.id"
            class="history-item"
            :class="{ active: conversation.id === sessionId }"
          >
            <button
              class="history-item-content"
              type="button"
              @click="openConversation(conversation)"
            >
              {{ conversation.title }}
            </button>
            <button
              class="history-delete-btn"
              type="button"
              title="删除此对话"
              @click.stop="deleteConversation(conversation)"
            >
              ×
            </button>
          </div>
        </div>
        <p v-else class="empty-history">暂无历史对话</p>
      </section>
    </aside>

    <aside class="map-drawer" :class="{ open: mapDrawerOpen }">
      <header class="map-drawer-header">
        <div>
          <h2>地图</h2>
          <p>{{ mapNotice }}</p>
        </div>
        <div class="map-layer-toggle" role="group" aria-label="地图图层">
          <button
            type="button"
            :class="{ active: mapViewMode === 'standard' }"
            @click="setMapViewMode('standard')"
          >
            标准
          </button>
          <button
            type="button"
            :class="{ active: mapViewMode === 'satellite' }"
            @click="setMapViewMode('satellite')"
          >
            卫星
          </button>
        </div>
        <button class="map-close-button" type="button" @click="mapDrawerOpen = false">×</button>
      </header>

      <div class="map-panel">
        <div v-if="mapLoading" class="map-state">地图加载中...</div>
        <div v-else-if="mapError" class="map-state error">{{ mapError }}</div>
        <div v-show="!mapLoading && !mapError" ref="arcgisContainerRef" class="arcgis-container" />
      </div>

      <footer class="map-drawer-footer">
        <span>灾害检测智能体</span>
      </footer>
    </aside>

    <section class="chat-pane">
      <img class="watermark" :src="watermarkUrl" alt="" aria-hidden="true" />

      <header class="chat-header">
        <div>
          <h1>{{ currentConversationTitle }}</h1>
          <p>{{ currentConversationUpdatedAt }}</p>
        </div>
        <div class="header-actions">
          <span class="header-status">{{ isSending ? '分析中' : '就绪' }}</span>
          <button
            class="map-toggle"
            type="button"
            :aria-label="mapDrawerOpen ? '隐藏地图' : '显示地图'"
            @click="mapDrawerOpen ? (mapDrawerOpen = false) : openMapDrawer()"
          >
            {{ mapDrawerOpen ? '隐藏地图' : '地图' }}
          </button>
        </div>
      </header>

      <div ref="chatContentRef" class="chat-content" @scroll="updateScrollBottomButton">
        <section v-if="!hasMessages" class="empty-state">
          <h1>What can I help analyze?</h1>
          <p>Ask about disaster damage, flood inundation, indices, statistics, or attach files below.</p>
        </section>

        <article
          v-for="message in messages"
          :key="message.id"
          class="message-row"
          :class="message.role"
        >
          <div class="avatar">{{ message.role === 'user' ? '☻' : '▣' }}</div>
          <div class="message-body">
            <div class="markdown-body" v-html="renderMarkdown(message.content)" />
            <div v-if="message.attachments?.length" class="attachment-list">
              <figure
                v-for="file in message.attachments"
                :key="file.id"
                class="attachment"
                :class="{ image: file.preview }"
              >
                <img v-if="file.preview" :src="file.preview" :alt="file.name" />
                <a
                  v-else-if="file.url"
                  class="file-chip downloadable"
                  :href="file.url"
                  :download="file.name"
                  target="_blank"
                  rel="noreferrer"
                >
                  <Document />
                  <strong>{{ fileExtension(file.name) }}</strong>
                  <span>下载</span>
                </a>
                <div v-else class="file-chip">
                  <Document />
                  <strong>{{ fileExtension(file.name) }}</strong>
                </div>
                <figcaption :title="file.name">
                  {{ file.url ? outputFileLabel(file.name) : file.name }}
                </figcaption>
              </figure>
            </div>
            <span v-if="message.meta" class="message-meta">{{ message.meta }}</span>
            <div v-if="message.images?.length" class="result-images">
              <figure
                v-for="(image, imageIndex) in message.images"
                :key="image.url"
                class="result-image"
              >
                <img :src="image.url" :alt="image.name" />
                <figcaption :title="image.name">{{ resultImageCaption(image) }}</figcaption>
                <div v-if="imageLegendItems(image, message).length" class="result-legend image-legend">
                  <span
                    v-for="item in imageLegendItems(image, message)"
                    :key="`${item.label}-${item.color}`"
                    class="legend-item"
                  >
                    <i :style="{ backgroundColor: item.color }" />
                    <span>{{ item.label }}</span>
                  </span>
                </div>
              </figure>
            </div>
            <div v-if="message.report" class="report-card">
              <div class="report-card-icon">
                <Document />
              </div>
              <div class="report-card-body">
                <div class="report-card-title">评估报告 (PDF)</div>
                <p class="report-card-desc">{{ message.report.description }}</p>
              </div>
              <div class="report-card-actions">
                <button
                  type="button"
                  class="report-btn preview-btn"
                  @click="openPdfPreview(message.report)"
                >
                  预览
                </button>
                <a
                  class="report-btn download-btn"
                  :href="message.report.url"
                  :download="message.report.name"
                  target="_blank"
                  rel="noreferrer"
                >
                  下载
                </a>
              </div>
            </div>
          </div>
        </article>
      </div>

      <button
        v-if="showScrollBottom"
        class="scroll-bottom-button"
        type="button"
        title="回到底部"
        @click="scrollToBottom('smooth')"
      >
        <Bottom />
      </button>

      <form class="composer" @submit.prevent="sendMessage">
        <label class="upload-button" title="添加文件">
          <Plus />
          <input ref="fileInputRef" multiple type="file" @change="handleFiles" />
        </label>

        <div class="composer-main">
          <div v-if="attachments.length" class="pending-files">
            <button
              v-for="file in attachments"
              :key="file.id"
              type="button"
              class="pending-file"
              @click="clearAttachment(file.id)"
            >
              <span>{{ file.name }}</span>
              <small>{{ formatSize(file.size) }}</small>
            </button>
          </div>
          <textarea
            v-model="inputText"
            placeholder="输入问题，或添加栅格/图片文件；时序数据请按时间先后顺序上传"
            rows="1"
            :disabled="isSending"
            @keydown.enter.exact.prevent="sendMessage"
          />
        </div>

        <button class="send-button" type="submit" title="Send" :disabled="isSending">
          <Top />
        </button>
      </form>
    </section>

    <div v-if="pdfPreviewVisible" class="pdf-preview-overlay" @click.self="closePdfPreview">
      <div class="pdf-preview-modal">
        <header class="pdf-preview-header">
          <span class="pdf-preview-name">{{ pdfPreviewName }}</span>
          <div class="pdf-preview-actions">
            <a
              class="pdf-preview-download"
              :href="pdfPreviewUrl"
              :download="pdfPreviewName"
              target="_blank"
              rel="noreferrer"
            >
              下载
            </a>
            <button type="button" class="pdf-preview-close" @click="closePdfPreview">×</button>
          </div>
        </header>
        <div class="pdf-preview-content">
          <iframe :src="pdfPreviewUrl" :title="pdfPreviewName" />
        </div>
      </div>
    </div>
  </main>
</template>
