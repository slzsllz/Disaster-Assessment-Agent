<script setup>
import { computed, ref } from 'vue'
import logoUrl from '../assets/szu-logo.png'
import { apiJson } from '../api'

defineProps({ notice: { type: String, default: '' } })
const emit = defineEmits(['authenticated'])
const mode = ref('login')
const username = ref('')
const displayName = ref('')
const password = ref('')
const confirmation = ref('')
const showPassword = ref(false)
const busy = ref(false)
const error = ref('')
const registering = computed(() => mode.value === 'register')

function switchMode(next) {
  mode.value = next
  password.value = ''
  confirmation.value = ''
  showPassword.value = false
  error.value = ''
}

async function submit() {
  if (busy.value) return
  error.value = ''
  if (registering.value && password.value !== confirmation.value) {
    error.value = '两次输入的密码不一致。'
    return
  }
  busy.value = true
  try {
    const body = { username: username.value.trim(), password: password.value }
    if (registering.value) body.display_name = displayName.value.trim()
    const data = await apiJson(`/api/auth/${mode.value}`, { method: 'POST', body }, false)
    password.value = ''
    confirmation.value = ''
    emit('authenticated', data.user)
  } catch (err) {
    error.value = err.status ? err.message : '无法连接服务，请稍后重试。'
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <main class="auth-page">
    <section class="auth-intro">
      <img :src="logoUrl" alt="深圳大学" class="auth-logo" />
      <span class="auth-eyebrow">DISASTER ASSESSMENT AGENT</span>
      <h1>灾害检测助手</h1>
      <p>从遥感影像出发，分析灾害影响。<br />在专属工作空间中保存对话、评估结果与报告。</p>
      <div class="auth-features"><span>遥感分析</span><span>空间评估</span><span>报告生成</span></div>
    </section>
    <section class="auth-card" aria-labelledby="auth-heading">
      <div class="auth-tabs" role="group" aria-label="登录或注册">
        <button type="button" :class="{ active: !registering }" :aria-pressed="!registering" :disabled="busy" @click="switchMode('login')">登录</button>
        <button type="button" :class="{ active: registering }" :aria-pressed="registering" :disabled="busy" @click="switchMode('register')">注册</button>
      </div>
      <h2 id="auth-heading">{{ registering ? '创建你的账号' : '欢迎回来' }}</h2>
      <p class="auth-subtitle">{{ registering ? '注册后即可开始分析，所有记录仅对你可见。' : '登录后继续你的灾害分析工作。' }}</p>
      <p v-if="notice" class="auth-notice" role="status">{{ notice }}</p>
      <form class="account-form" @submit.prevent="submit">
        <fieldset :disabled="busy">
          <label for="auth-username">用户名</label>
          <input id="auth-username" v-model="username" name="username" autocomplete="username" required minlength="3" maxlength="32"
            pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{2,31}" placeholder="输入用户名" autocapitalize="none" spellcheck="false" />
          <small v-if="registering" class="field-hint">3–32 位字母、数字、下划线、点或短横线，不区分大小写。</small>
          <template v-if="registering">
            <label for="auth-nickname">昵称 <span class="field-optional">选填</span></label>
            <input id="auth-nickname" v-model="displayName" name="nickname" autocomplete="nickname" maxlength="50" placeholder="你的显示名称" />
          </template>
          <label for="auth-password">密码</label>
          <div class="password-input">
            <input id="auth-password" v-model="password" name="password" :type="showPassword ? 'text' : 'password'"
              :autocomplete="registering ? 'new-password' : 'current-password'" :minlength="registering ? 12 : 1" maxlength="128" required
              :placeholder="registering ? '至少 12 个字符，支持中文和空格' : '输入密码'" />
            <button type="button" :aria-label="showPassword ? '隐藏密码' : '显示密码'" @click="showPassword = !showPassword">{{ showPassword ? '隐藏' : '显示' }}</button>
          </div>
          <template v-if="registering">
            <label for="auth-confirm">确认密码</label>
            <input id="auth-confirm" v-model="confirmation" name="password-confirm" :type="showPassword ? 'text' : 'password'"
              autocomplete="new-password" required minlength="12" maxlength="128" placeholder="再次输入密码" />
          </template>
          <p v-if="error" class="form-error" role="alert">{{ error }}</p>
          <button class="account-primary" type="submit">{{ busy ? '请稍候…' : registering ? '注册并进入工作空间' : '登录' }}</button>
        </fieldset>
      </form>
    </section>
  </main>
</template>
