<script setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { videoPreviewUrl } from '../utils/mediaFiles.js'

const props = defineProps({
  src: { type: String, default: '' },
  name: { type: String, default: '视频' },
  downloadUrl: { type: String, default: '' },
  active: { type: Boolean, default: true },
  original: { type: Boolean, default: true },
})
const video = ref(null)
const failed = ref(false)
const compatible = ref(false)
const preparing = ref(false)
const previewUrl = computed(() => videoPreviewUrl(props.downloadUrl))
const source = computed(() => compatible.value ? previewUrl.value : props.src)

function reset() { failed.value = false; compatible.value = false; preparing.value = false }
function onError() { failed.value = true; preparing.value = false }
async function preparePreview() {
  failed.value = false
  compatible.value = true
  preparing.value = true
  await nextTick()
  // load() also retries a previously failed compatibility URL.
  video.value?.load()
}
watch(() => props.src, reset)
watch(() => props.active, active => { if (!active) video.value?.pause() })
onBeforeUnmount(() => {
  if (!video.value) return
  video.value.pause()
  video.value.removeAttribute('src')
  video.value.load()
})
</script>

<template>
  <div class="video-attachment">
    <video v-if="source" ref="video" :src="source" :aria-label="name" controls playsinline preload="metadata"
      @error="onError" @loadedmetadata="preparing = false; failed = false">
      当前浏览器不支持视频播放，请下载视频。
    </video>
    <p v-if="preparing" class="video-hint" role="status">正在准备兼容预览，请稍候...</p>
    <p v-if="failed" class="video-hint" role="alert">
      {{ compatible ? '兼容预览暂时无法播放，可重试或下载视频。' : '当前浏览器无法播放此视频。' }}
      <button v-if="previewUrl" type="button" @click="preparePreview">{{ compatible ? '重试预览' : '生成兼容预览' }}</button>
      <span v-else>可发送后生成兼容预览，或下载视频。</span>
    </p>
    <p v-else-if="!source" class="video-hint">视频链接不可用。</p>
    <a v-if="downloadUrl || src" class="video-download" :href="downloadUrl || src" :download="name">{{ original ? '下载原视频' : '下载视频' }}</a>
  </div>
</template>

<style scoped>
.video-attachment { width: 100%; min-width: 0; }
video { display: block; width: 100%; max-height: 360px; aspect-ratio: 16 / 9; object-fit: contain; background: #111827; border-radius: 10px; }
.video-hint { margin: 8px 0; font-size: 12px; line-height: 1.6; color: #6b7280; }
.video-hint button { margin-left: 6px; padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 5px; background: white; cursor: pointer; color: #334155; }
.video-download { display: inline-block; margin-top: 6px; color: #3b6091; font-size: 12px; text-decoration: none; }
.video-download:hover { text-decoration: underline; }
</style>
