<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { apiJson } from '../api'

const props = defineProps({ user: { type: Object, required: true } })
const emit = defineEmits(['close', 'updated', 'password-changed'])
const dialogRef = ref(null)
const displayName = ref(props.user.display_name)
const currentPassword = ref('')
const newPassword = ref('')
const confirmation = ref('')
const busy = ref(false)
const message = ref('')
const error = ref('')
let previousFocus = null

function close() { if (!busy.value) emit('close') }

async function saveProfile() {
  if (busy.value) return
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    const data = await apiJson('/api/auth/me', { method: 'PATCH', body: { display_name: displayName.value.trim() } })
    emit('updated', data.user)
    message.value = '昵称已更新。'
  } catch (err) {
    error.value = err.status ? err.message : '无法连接服务，请稍后重试。'
  } finally { busy.value = false }
}

async function changePassword() {
  if (busy.value) return
  error.value = ''
  message.value = ''
  if (newPassword.value !== confirmation.value) {
    error.value = '两次输入的新密码不一致。'
    return
  }
  busy.value = true
  try {
    await apiJson('/api/auth/password', { method: 'POST', body: {
      current_password: currentPassword.value, new_password: newPassword.value,
    } })
    currentPassword.value = ''
    newPassword.value = ''
    confirmation.value = ''
    emit('password-changed')
  } catch (err) {
    error.value = err.status ? err.message : '无法连接服务，请稍后重试。'
  } finally { busy.value = false }
}

onMounted(() => {
  previousFocus = document.activeElement
  dialogRef.value.showModal()
  dialogRef.value.querySelector('input')?.focus()
})
onUnmounted(() => previousFocus?.focus())
</script>

<template>
  <dialog ref="dialogRef" class="account-dialog" aria-labelledby="account-heading" @cancel.prevent="close" @click="($event.target === dialogRef) && close()">
    <header class="account-dialog-header"><h2 id="account-heading">账号设置</h2><button type="button" :disabled="busy" @click="close" aria-label="关闭账号设置">×</button></header>
    <p class="account-username">用户名：{{ user.username }}</p>
    <p v-if="message" class="auth-notice" role="status">{{ message }}</p>
    <p v-if="error" class="form-error" role="alert">{{ error }}</p>
    <form class="account-form" @submit.prevent="saveProfile">
      <fieldset :disabled="busy"><legend>个人资料</legend>
        <label for="profile-nickname">昵称</label>
        <input id="profile-nickname" v-model="displayName" autocomplete="nickname" required maxlength="50" />
        <button class="account-primary" type="submit">保存昵称</button>
      </fieldset>
    </form>
    <form class="account-form password-change-form" @submit.prevent="changePassword">
      <fieldset :disabled="busy"><legend>修改密码</legend>
        <p class="field-hint">修改后所有设备都会退出登录，请使用新密码重新登录。</p>
        <label for="password-current">当前密码</label>
        <input id="password-current" v-model="currentPassword" type="password" autocomplete="current-password" required maxlength="128" />
        <label for="password-new">新密码</label>
        <input id="password-new" v-model="newPassword" type="password" autocomplete="new-password" required minlength="12" maxlength="128" placeholder="至少 12 个字符" />
        <label for="password-confirm">确认新密码</label>
        <input id="password-confirm" v-model="confirmation" type="password" autocomplete="new-password" required minlength="12" maxlength="128" />
        <button class="account-primary" type="submit">{{ busy ? '请稍候…' : '修改密码并重新登录' }}</button>
      </fieldset>
    </form>
  </dialog>
</template>
