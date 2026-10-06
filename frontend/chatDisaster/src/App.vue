<script setup>
import { defineAsyncComponent, onMounted, onUnmounted, ref } from 'vue'
import AuthPanel from './components/AuthPanel.vue'
import AccountDialog from './components/AccountDialog.vue'
import { apiJson } from './api'

const ChatWorkspace = defineAsyncComponent(() => import('./ChatWorkspace.vue'))
const user = ref(null)
const loading = ref(true)
const loggingOut = ref(false)
const accountOpen = ref(false)
const notice = ref('')
const connectionError = ref('')

function clearSessionUrl() {
  const url = new URL(window.location.href)
  url.searchParams.delete('session_id')
  window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`)
}

function signedOut(message = '') {
  user.value = null
  accountOpen.value = false
  notice.value = message
  clearSessionUrl()
}

function sessionExpired() {
  if (user.value) signedOut('登录已过期，请重新登录。')
}

function authenticated(account) {
  user.value = account
  notice.value = ''
  connectionError.value = ''
}

async function restoreLogin() {
  loading.value = true
  connectionError.value = ''
  try {
    const data = await apiJson('/api/auth/me', {}, false)
    user.value = data.user
  } catch (error) {
    if (error.status !== 401) connectionError.value = error.status ? error.message : '无法连接服务，请检查后端是否已启动。'
  } finally {
    loading.value = false
  }
}

async function logout() {
  if (loggingOut.value) return
  loggingOut.value = true
  try {
    await apiJson('/api/auth/logout', { method: 'POST' })
    signedOut('已退出登录。')
  } catch (error) {
    connectionError.value = error.status ? error.message : '退出失败，请检查网络后重试。'
  } finally {
    loggingOut.value = false
  }
}

onMounted(() => {
  window.addEventListener('auth-expired', sessionExpired)
  restoreLogin()
})
onUnmounted(() => window.removeEventListener('auth-expired', sessionExpired))
</script>

<template>
  <div v-if="loading" class="auth-loading" role="status">正在恢复登录状态…</div>
  <template v-else>
    <div v-if="connectionError" class="connection-banner" role="alert">
      <span>{{ connectionError }}</span>
      <button v-if="!user" type="button" @click="restoreLogin">重新连接</button>
      <button v-else type="button" @click="connectionError = ''" aria-label="关闭提示">×</button>
    </div>
    <ChatWorkspace v-if="user" :key="user.id" :user="user" :logging-out="loggingOut"
      @account="accountOpen = true" @logout="logout" />
    <AuthPanel v-else :notice="notice" @authenticated="authenticated" />
    <AccountDialog v-if="user && accountOpen" :user="user" @close="accountOpen = false"
      @updated="user = $event" @password-changed="signedOut('密码已修改，请使用新密码登录。')" />
  </template>
</template>
