<script setup>
import { Bottom, Document, Plus, Top } from '@element-plus/icons-vue'
import { computed, nextTick, ref } from 'vue'
import logoUrl from './assets/szu-logo.png'

const promptMode = ref('default')
const systemPrompt = ref(
  "You are a geoscientist, and you need to use tools to answer Earth observation questions. Carefully reason about which tools to use and in what order. When a tool returns 'Result saved at /path/to/file', you MUST use that full path in all subsequent tool calls. Finish your final response with a clearly labelled answer block."
)
const recursionLimit = ref(40)
const maxExecutionTime = ref(600)
const showTrace = ref(false)
const inputText = ref('')
const attachments = ref([])
const messages = ref([])
const sidebarOpen = ref(true)
const isSending = ref(false)
const sessionId = ref(localStorage.getItem('chatDisasterSessionId') || crypto.randomUUID())
const chatContentRef = ref(null)
const showScrollBottom = ref(false)

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
  }
}

async function clearHistory() {
  messages.value = []
  showScrollBottom.value = false
  attachments.value.forEach((item) => {
    if (item.preview) URL.revokeObjectURL(item.preview)
  })
  attachments.value = []
  try {
    await fetch(`/api/sessions/${sessionId.value}/clear`, { method: 'POST' })
  } catch {
    // UI state is already cleared; backend can be reset on next refresh if needed.
  }
}
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
        <label class="section-label">System prompt</label>
        <label class="radio-row">
          <input v-model="promptMode" value="default" type="radio" />
          <span>Default (concise)</span>
        </label>
        <label class="radio-row">
          <input v-model="promptMode" value="custom" type="radio" />
          <span>Custom</span>
        </label>
      </section>

      <section class="sidebar-section">
        <label class="section-label" for="system-prompt">System prompt</label>
        <textarea id="system-prompt" v-model="systemPrompt" class="system-textarea" />
      </section>

      <section class="sidebar-section">
        <h2>Advanced</h2>
        <label class="slider-label">
          <span>Recursion limit</span>
          <strong>{{ recursionLimit }}</strong>
        </label>
        <input v-model="recursionLimit" type="range" min="10" max="100" step="5" />

        <label class="slider-label">
          <span>Max execution time (s)</span>
          <strong>{{ maxExecutionTime }}</strong>
        </label>
        <input v-model="maxExecutionTime" type="range" min="60" max="1800" step="60" />

        <label class="checkbox-row">
          <input v-model="showTrace" type="checkbox" />
          <span>Show tool-call trace</span>
        </label>
      </section>

      <button class="clear-button" type="button" @click="clearHistory">Clear chat history</button>
      <p class="ready-text">Ready · 2 tools loaded</p>
    </aside>

    <section class="chat-pane">
      <img class="watermark" :src="logoUrl" alt="" aria-hidden="true" />

      <header class="chat-header">
        <div>
          <h1>哈哈哈哈哈哈哈哈哈哈哈哈哈哈</h1>
          <p>占位 占位 占位 占位 占位 占位 占位 占位 占位</p>
        </div>
        <span class="header-status">{{ isSending ? 'Analyzing' : 'Ready' }}</span>
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
