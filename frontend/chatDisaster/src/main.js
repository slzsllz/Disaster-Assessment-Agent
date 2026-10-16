import { createApp } from 'vue'
import './style.css'
import App from './App.vue'

if (!globalThis.crypto) {
  globalThis.crypto = {}
}
if (!globalThis.crypto.randomUUID) {
  globalThis.crypto.randomUUID = function() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const random = new Uint8Array(1)
      const r = globalThis.crypto.getRandomValues
        ? globalThis.crypto.getRandomValues(random)[0] & 15
        : Math.random() * 16 | 0
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }
}

createApp(App).mount('#app')
