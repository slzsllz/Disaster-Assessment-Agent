<script setup>
import { Bottom, Document, Edit, Plus, Top } from '@element-plus/icons-vue'
import { ElSlider } from 'element-plus'
import 'element-plus/es/components/slider/style/css'
import { computed, nextTick, onMounted, ref } from 'vue'
import logoUrl from './assets/szu-logo.png'

const systemPrompt = ref(
  "You are a geoscientist, and you need to use tools to answer Earth observation questions. Carefully reason about which tools to use and in what order. When a tool returns 'Result saved at /path/to/file', you MUST use that full path in all subsequent tool calls. Finish your final response with a clearly labelled answer block."
)
const recursionLimit = ref(40)
const maxExecutionTime = ref(600)
const showTrace = ref(false)
const inputText = ref('')
const attachments = ref([])
const sidebarOpen = ref(true)
const mapDrawerOpen = ref(false)
const isSending = ref(false)
const HISTORY_KEY = 'chatDisasterConversations'

function loadConversationHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

const conversationHistory = ref(loadConversationHistory())
const sessionId = ref(localStorage.getItem('chatDisasterSessionId') || crypto.randomUUID())
const savedConversation = conversationHistory.value.find((item) => item.id === sessionId.value)
const messages = ref(savedConversation?.messages || [])
const chatContentRef = ref(null)
const showScrollBottom = ref(false)
const amapContainerRef = ref(null)
const amapMap = ref(null)
const amapLoading = ref(false)
const amapError = ref('')
const mapViewMode = ref('standard')
const amapKey = import.meta.env.VITE_AMAP_KEY || ''
const amapSecurityCode = import.meta.env.VITE_AMAP_SECURITY_CODE || ''

localStorage.setItem('chatDisasterSessionId', sessionId.value)

const hasMessages = computed(() => messages.value.length > 0)
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

function renderMarkdown(value) {
  const lines = String(value ?? '').split('\n')
  const html = []
  let listOpen = false

  function closeList() {
    if (listOpen) {
      html.push('</ul>')
      listOpen = false
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) {
      closeList()
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

function resultImageCaption(index) {
  return index > 0 ? `可视化结果 ${index + 1}` : '可视化结果'
}

function conversationTitle(items = messages.value) {
  const firstUserMessage = items.find((item) => item.role === 'user')?.content || '新对话'
  const title = firstUserMessage.replace(/\s+/g, ' ').trim() || '新对话'
  return title.length > 24 ? `${title.slice(0, 24)}...` : title
}

function sanitizeMessageForHistory(message) {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    meta: message.meta || '',
    error: message.error || '',
    images: message.images || [],
    attachments: (message.attachments || []).map((file) => ({
      id: file.id,
      name: file.name,
      size: file.size,
      type: file.type,
      preview: '',
    })),
  }
}

function saveConversationHistory() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(conversationHistory.value))
}

// ---------------------------------------------------------------------------
// 后端数据加载 -- 会话历史与消息从数据库读取 (DB -> API -> 前端)
// 失败时静默回退到 localStorage, 不阻断本地使用。
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
    attachments: (m.attachments || []).map((a) => ({
      id: a.name || a.path || 'file',
      name: a.name || 'file',
      size: a.size || 0,
      type: a.type || '',
      preview: '',
    })),
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
    const apiSessions = (data.sessions || []).map(mapApiSession)
    const apiIds = new Set(apiSessions.map((s) => s.id))
    // 保留 localStorage 中 DB 尚未收录的会话 (离线/旧数据)
    const localOnly = conversationHistory.value.filter((c) => !apiIds.has(c.id))
    conversationHistory.value = [...apiSessions, ...localOnly].slice(0, 30)
    saveConversationHistory()
  } catch {
    // DB 不可用, 保持 localStorage 历史
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

function upsertCurrentConversation() {
  if (!messages.value.length) return

  const item = {
    id: sessionId.value,
    title: conversationTitle(),
    updatedAt: Date.now(),
    messages: messages.value.map(sanitizeMessageForHistory),
  }
  conversationHistory.value = [
    item,
    ...conversationHistory.value.filter((historyItem) => historyItem.id !== item.id),
  ].slice(0, 30)
  saveConversationHistory()
}

function releasePendingFiles() {
  attachments.value.forEach((item) => {
    if (item.preview) URL.revokeObjectURL(item.preview)
  })
  attachments.value = []
}

function startNewConversation() {
  upsertCurrentConversation()
  releasePendingFiles()
  messages.value = []
  showScrollBottom.value = false
  sessionId.value = crypto.randomUUID()
  localStorage.setItem('chatDisasterSessionId', sessionId.value)
}

async function openConversation(conversation) {
  if (conversation.id === sessionId.value) return
  upsertCurrentConversation()
  releasePendingFiles()
  sessionId.value = conversation.id
  localStorage.setItem('chatDisasterSessionId', sessionId.value)
  // 优先用内存缓存的消息; 否则从后端 (DB) 拉取
  if (conversation.messages && conversation.messages.length) {
    messages.value = conversation.messages
  } else {
    messages.value = []
    const fetched = await fetchMessages(conversation.id)
    if (fetched && fetched.length) {
      messages.value = fetched
      conversation.messages = fetched
    }
  }
  showScrollBottom.value = false
  scrollToBottom()
}

function applyAmapLayer() {
  if (!amapMap.value || !window.AMap) return
  const AMap = window.AMap
  if (mapViewMode.value === 'satellite') {
    amapMap.value.setLayers([new AMap.TileLayer.Satellite()])
  } else {
    amapMap.value.setLayers([new AMap.TileLayer()])
  }
}

function setMapViewMode(mode) {
  mapViewMode.value = mode
  applyAmapLayer()
}

function loadAmapScript() {
  if (window.AMap) return Promise.resolve(window.AMap)
  if (!amapKey) return Promise.reject(new Error('缺少 VITE_AMAP_KEY，请在前端环境变量中配置高德 Web JS API Key。'))

  if (amapSecurityCode) {
    window._AMapSecurityConfig = {
      securityJsCode: amapSecurityCode,
    }
  }

  const existingScript = document.querySelector('script[data-amap-sdk="true"]')
  if (existingScript) {
    return new Promise((resolve, reject) => {
      existingScript.addEventListener('load', () => resolve(window.AMap), { once: true })
      existingScript.addEventListener('error', () => reject(new Error('高德地图 SDK 加载失败。')), {
        once: true,
      })
    })
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.dataset.amapSdk = 'true'
    script.src = `https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(amapKey)}&plugin=AMap.Scale,AMap.ToolBar`
    script.async = true
    script.onload = () => resolve(window.AMap)
    script.onerror = () => reject(new Error('高德地图 SDK 加载失败，请检查网络或 Key 配置。'))
    document.head.appendChild(script)
  })
}

async function initAmap() {
  if (amapMap.value || amapLoading.value) return

  amapLoading.value = true
  amapError.value = ''
  try {
    const AMap = await loadAmapScript()
    await nextTick()
    if (!amapContainerRef.value) return

    amapMap.value = new AMap.Map(amapContainerRef.value, {
      zoom: 4,
      center: [104.1954, 35.8617],
      viewMode: '2D',
      resizeEnable: true,
    })
    amapMap.value.addControl(new AMap.Scale())
    amapMap.value.addControl(new AMap.ToolBar({ position: 'RB' }))
    applyAmapLayer()
  } catch (error) {
    amapError.value = error.message || String(error)
  } finally {
    amapLoading.value = false
  }
}

async function openMapDrawer() {
  mapDrawerOpen.value = true
  await nextTick()
  await initAmap()
  nextTick(() => {
    amapMap.value?.resize()
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
    message.error = ''
  } else if (eventName === 'error') {
    message.content = payload.answer || '后端调用失败'
    message.meta = ''
    message.images = []
    message.error = payload.error || ''
  }
  followBottomIfNeeded()
}

async function sendMessage() {
  if (isSending.value) return
  const text = inputText.value.trim()
  if (!text && attachments.value.length === 0) return

  const localAttachments = [...attachments.value]
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
    upsertCurrentConversation()
  }
}

onMounted(async () => {
  // 当前会话若内存无消息 (如新浏览器/刷新), 从后端 DB 加载
  if (!messages.value.length) {
    const fetched = await fetchMessages(sessionId.value)
    if (fetched && fetched.length) {
      messages.value = fetched
      const saved = conversationHistory.value.find((c) => c.id === sessionId.value)
      if (saved) saved.messages = fetched
      scrollToBottom()
    }
  }
  // 用 DB 会话刷新历史侧栏 (DB 不可用时回退 localStorage)
  await fetchSessions()
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
      <div class="sidebar-title">Disaster Detection Agent</div>

      <section class="sidebar-section">
        <h2>Advanced</h2>
        <label class="slider-label">
          <span>Recursion limit</span>
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
          <span>Max execution time (s)</span>
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
          <span>Show tool-call trace</span>
        </label>
      </section>

      <button class="new-chat-button" type="button" @click="startNewConversation">
        <Edit />
        <span>创建新对话</span>
      </button>

      <section class="history-section">
        <h2>历史对话</h2>
        <div v-if="conversationHistory.length" class="history-list">
          <button
            v-for="conversation in conversationHistory"
            :key="conversation.id"
            class="history-item"
            :class="{ active: conversation.id === sessionId }"
            type="button"
            @click="openConversation(conversation)"
          >
            {{ conversation.title }}
          </button>
        </div>
        <p v-else class="empty-history">暂无历史对话</p>
      </section>
    </aside>

    <aside class="map-drawer" :class="{ open: mapDrawerOpen }">
      <header class="map-drawer-header">
        <div>
          <h2>地图</h2>
        </div>
        <div class="map-layer-toggle" role="group" aria-label="Map layer">
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
        <div v-if="amapLoading" class="map-state">Loading AMap...</div>
        <div v-else-if="amapError" class="map-state error">{{ amapError }}</div>
        <div v-show="!amapLoading && !amapError" ref="amapContainerRef" class="amap-container" />
      </div>

      <footer class="map-drawer-footer">
        <span>Disaster Detection Agent</span>
      </footer>
    </aside>

    <section class="chat-pane">
      <img class="watermark" :src="logoUrl" alt="" aria-hidden="true" />

      <header class="chat-header">
        <div>
          <h1>灾害遥感智能评估助手</h1>
          <p>基于多模态大模型与遥感 AI 工具的灾害损毁评估 · 洪水淹没 / 建筑损毁 / 火烧迹地</p>
        </div>
        <div class="header-actions">
          <span class="header-status">{{ isSending ? 'Analyzing' : 'Ready' }}</span>
          <button
            class="map-toggle"
            type="button"
            :aria-label="mapDrawerOpen ? 'Hide map' : 'Show map'"
            @click="mapDrawerOpen ? (mapDrawerOpen = false) : openMapDrawer()"
          >
            {{ mapDrawerOpen ? 'Hide map' : 'Map' }}
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
                <div v-else class="file-chip">
                  <Document />
                  <strong>{{ fileExtension(file.name) }}</strong>
                </div>
                <figcaption>{{ file.name }}</figcaption>
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
                <figcaption>{{ resultImageCaption(imageIndex) }}</figcaption>
              </figure>
            </div>
          </div>
        </article>
      </div>

      <button
        v-if="showScrollBottom"
        class="scroll-bottom-button"
        type="button"
        title="Back to bottom"
        @click="scrollToBottom('smooth')"
      >
        <Bottom />
      </button>

      <form class="composer" @submit.prevent="sendMessage">
        <label class="upload-button" title="Attach files">
          <Plus />
          <input multiple type="file" @change="handleFiles" />
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
            placeholder="Ask anything, or attach raster/image files"
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
  </main>
</template>
