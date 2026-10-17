<script setup>
import { nextTick, ref, watch } from 'vue'

const props = defineProps({
  open: Boolean,
  news: { type: Array, default: () => [] },
  lastChecked: { type: String, default: '' },
})
const emit = defineEmits(['update:open'])
const dialog = ref(null)

watch([() => props.open, dialog], async ([open, element]) => {
  await nextTick()
  if (!element) return
  if (open && !element.open) element.showModal()
  if (!open && element.open) element.close()
}, { flush: 'post' })

function close() { emit('update:open', false) }
function formatTime(value) {
  if (!value) return '发布时间未知'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}
</script>

<template>
  <Teleport to="body">
    <dialog ref="dialog" class="disaster-dialog" aria-labelledby="disaster-news-title" @cancel.prevent="close" @close="close" @click.self="close">
      <section class="disaster-dialog-body">
        <header class="disaster-dialog-header">
          <div>
            <span class="news-eyebrow">灾情动态 · 网络报道</span>
            <h2 id="disaster-news-title">最新灾情消息 <span>{{ news.length }}</span></h2>
          </div>
          <button class="news-close" type="button" aria-label="关闭灾情消息弹窗" @click="close">×</button>
        </header>
        <p class="news-explanation">每分钟检索一次，优先关注官方通报。以下为近期网络报道，请以原始来源及官方最新信息为准。</p>
        <div class="news-list">
          <article v-for="item in news" :key="item.id" class="news-card">
            <div class="news-meta"><span>{{ item.source }}</span><time :datetime="item.published_at">{{ formatTime(item.published_at) }}</time></div>
            <h3><a :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title }} <span aria-hidden="true">↗</span></a></h3>
            <p>{{ item.summary || '打开原始报道查看详情。' }}</p>
          </article>
          <p v-if="!news.length" class="news-empty">暂未检索到新的灾情报道，正在持续关注。</p>
        </div>
        <footer class="news-footer">
          <span>{{ lastChecked ? `最近检索 ${formatTime(lastChecked)}` : '正在等待首次检索' }}</span>
          <button type="button" @click="close">我知道了</button>
        </footer>
      </section>
    </dialog>
  </Teleport>
</template>

<style scoped>
.disaster-dialog { width: min(660px, calc(100vw - 32px)); max-height: min(78vh, 800px); padding: 0; border: 1px solid #e4e7ec; border-radius: 18px; color: #202123; background: white; box-shadow: 0 24px 80px #10182833; }
.disaster-dialog::backdrop { background: #11182766; backdrop-filter: blur(3px); }
.disaster-dialog-body { display: flex; flex-direction: column; max-height: min(78vh, 800px); }
.disaster-dialog-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; padding: 24px 24px 12px; }
.news-eyebrow { color: #a15c16; font-size: 12px; font-weight: 600; }
h2 { margin: 8px 0 0; font-size: 21px; }
h2 span { display: inline-block; margin-left: 5px; padding: 2px 9px; border-radius: 20px; background: #fff4e5; color: #9a580f; font-size: 13px; vertical-align: middle; }
.news-close { border: 0; border-radius: 8px; background: #f3f4f6; color: #667085; width: 30px; height: 30px; font-size: 24px; line-height: 1; }
.news-explanation { margin: 0; padding: 0 24px 18px; font-size: 12px; line-height: 1.7; color: #667085; }
.news-list { overflow-y: auto; min-height: 0; padding: 0 24px; }
.news-card { padding: 18px 0; border-top: 1px solid #eaecf0; }
.news-meta { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px 12px; font-size: 11px; color: #667085; }
.news-meta span { color: #8a5a22; font-weight: 600; }
h3 { margin: 10px 0 8px; line-height: 1.55; font-size: 15px; }
a { color: #202123; text-decoration: none; }
a:hover { color: #2563eb; text-decoration: underline; }
a span { color: #98a2b3; }
.news-card p { margin: 0; color: #667085; font-size: 13px; line-height: 1.8; white-space: pre-wrap; overflow-wrap: anywhere; }
.news-footer { padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; gap: 12px; border-top: 1px solid #eaecf0; }
.news-footer span { font-size: 11px; color: #667085; }
.news-footer button { flex-shrink: 0; border: 0; border-radius: 8px; padding: 9px 16px; background: #202123; color: white; font-size: 12px; }
.news-empty { padding: 28px 0; text-align: center; color: #667085; }
@media (max-width: 480px) { .disaster-dialog-header { padding: 18px 18px 12px; } .news-list { padding: 0 18px; } .news-explanation { padding: 0 18px 14px; } .news-footer { padding: 14px 18px; flex-wrap: wrap; } }
</style>
