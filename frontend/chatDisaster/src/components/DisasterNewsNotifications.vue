<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import DisasterNewsDialog from './DisasterNewsDialog.vue'

const enabled = ref(false)
const busy = ref(false)
const checking = ref(false)
const error = ref('')
const lastChecked = ref('')
const news = ref([])
const dialogOpen = ref(false)
const clientId = crypto.randomUUID()
let cursor = 0
let epoch = ''
let timer = null
let controller = null
let generation = 0
let disposed = false
let subscriptionRequest = null
let subscriptionRequested = false
const preferenceKey = 'disaster-news-enabled'
const seenKey = 'disaster-news-seen'
let seen = new Set()
try { seen = new Set(JSON.parse(localStorage.getItem(seenKey) || '[]')) } catch { /* Storage may be unavailable. */ }

const statusLabel = computed(() => error.value ? '检索异常，正在重试' : checking.value ? '正在检索灾情' : lastChecked.value ? '每分钟检索灾情' : '等待首次检索')

function storePreference(value) {
  try { localStorage.setItem(preferenceKey, value ? 'true' : 'false') } catch { /* Best effort. */ }
}

async function api(path, options = {}) {
  const response = await fetch(`/api/disaster-news/${path}`, options)
  if (!response.ok) throw new Error('无法连接灾情通知服务，请稍后重试。')
  return response.json()
}

function consume(state) {
  cursor = state.cursor
  epoch = state.epoch
  checking.value = state.checking
  lastChecked.value = state.last_checked_at || ''
  error.value = state.error || ''
  const fresh = (state.news || []).filter(item => !seen.has(item.id))
  for (const item of fresh) seen.add(item.id)
  seen = new Set([...seen].slice(-500))
  try { localStorage.setItem(seenKey, JSON.stringify([...seen])) } catch { /* Best effort. */ }
  if (fresh.length) {
    const merged = new Map([...news.value, ...fresh].map(item => [item.id, item]))
    news.value = [...merged.values()].sort((a, b) => Date.parse(b.published_at) - Date.parse(a.published_at)).slice(0, 50)
    dialogOpen.value = true
  }
}

function stopPolling() {
  generation += 1
  clearTimeout(timer)
  controller?.abort()
  controller = null
}

async function poll(token = generation) {
  if (!enabled.value || disposed || token !== generation) return
  controller = new AbortController()
  const timeout = setTimeout(() => controller?.abort(), 15000)
  try {
    const params = new URLSearchParams({ client_id: clientId, after_seq: String(cursor), epoch })
    let state = await api(`updates?${params}`, { signal: controller.signal })
    if (!enabled.value || disposed || token !== generation) return
    if (!state.subscribed) {
      state = await api('subscriptions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: clientId, enabled: true }), signal: controller.signal,
      })
    }
    if (enabled.value && token === generation && !disposed) consume(state)
  } catch (err) {
    if (enabled.value && token === generation && !disposed) error.value = '连接暂时中断，将自动重试。'
  } finally {
    clearTimeout(timeout)
    if (enabled.value && token === generation && !disposed) timer = setTimeout(() => poll(token), 10000)
  }
}

async function setEnabled(value) {
  if (busy.value || disposed) return
  busy.value = true
  subscriptionRequested = value
  stopPolling()
  if (!value) {
    enabled.value = false
    dialogOpen.value = false
    storePreference(false)
  }
  const request = new AbortController()
  subscriptionRequest = request
  const timeout = setTimeout(() => request.abort(), 15000)
  try {
    const state = await api('subscriptions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: clientId, enabled: value }), signal: request.signal,
    })
    if (disposed) return
    enabled.value = value
    storePreference(value)
    error.value = ''
    if (value) {
      consume(state)
      void poll()
    }
  } catch (err) {
    if (disposed) return
    error.value = value ? '通知开启失败，请检查服务连接后重试。' : '本页通知已关闭，后台订阅将在连接恢复或超时后停止。'
  } finally {
    clearTimeout(timeout)
    subscriptionRequest = null
    busy.value = false
  }
}

function releaseSubscription() {
  stopPolling()
  subscriptionRequest?.abort()
  if (!enabled.value && !subscriptionRequested) return
  // keepalive also releases the lease when navigating away or closing the tab.
  void fetch('/api/disaster-news/subscriptions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ client_id: clientId, enabled: false }), keepalive: true,
  }).catch(() => {})
}
function restorePage(event) {
  if (event.persisted && enabled.value) void poll()
}

onMounted(() => {
  let preference = false
  try { preference = localStorage.getItem(preferenceKey) === 'true' } catch { /* Default off. */ }
  if (preference) void setEnabled(true)
  window.addEventListener('pagehide', releaseSubscription)
  window.addEventListener('pageshow', restorePage)
})
onUnmounted(() => {
  disposed = true
  releaseSubscription()
  window.removeEventListener('pagehide', releaseSubscription)
  window.removeEventListener('pageshow', restorePage)
})
</script>

<template>
  <div class="disaster-notifications">
    <button type="button" class="notification-toggle" :class="{ active: enabled }" :disabled="busy" :aria-pressed="enabled" :aria-label="enabled ? '关闭灾情通知' : '开启灾情通知'" :title="error || (enabled ? statusLabel : '开启后每分钟检索灾情新闻，并弹窗通知')" @click="setEnabled(!enabled)">
      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4" /></svg>
      <span>{{ busy ? '连接中' : '灾情通知' }}</span>
      <span class="notification-switch" aria-hidden="true"><span /></span>
    </button>
    <button v-if="enabled && news.length" class="news-history" type="button" title="查看已收到的灾情消息" aria-haspopup="dialog" @click="dialogOpen = true">{{ news.length }} 条</button>
    <span v-if="error" class="notification-error" role="status" :title="error">{{ enabled ? '重试中' : '连接异常' }}</span>
    <DisasterNewsDialog v-model:open="dialogOpen" :news="news" :last-checked="lastChecked" />
  </div>
</template>

<style scoped>
.disaster-notifications { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.notification-toggle { display: inline-flex; align-items: center; justify-content: center; gap: 6px; height: 32px; padding: 0 10px; border: 1px solid #d9d9df; border-radius: 999px; background: white; color: #667085; font-size: 12px; white-space: nowrap; }
.notification-toggle.active { color: #8b5312; border-color: #efcea0; background: #fffaf2; }
.notification-toggle:disabled { cursor: wait; opacity: .65; }
.notification-switch { position: relative; width: 24px; height: 14px; background: #d0d5dd; border-radius: 20px; transition: background .15s; }
.notification-switch span { position: absolute; top: 2px; left: 2px; width: 10px; height: 10px; background: white; border-radius: 50%; transition: transform .15s; }
.active .notification-switch { background: #b77825; }
.active .notification-switch span { transform: translateX(10px); }
.news-history { border: 0; padding: 3px 6px; border-radius: 5px; background: #fff4e5; color: #8b5312; font-size: 11px; white-space: nowrap; }
.notification-error { color: #b42318; font-size: 11px; }
</style>
