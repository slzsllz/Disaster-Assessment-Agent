import { createApp } from 'vue'
import './style.css'
import App from './App.vue'

// main.js
if (!globalThis.crypto) {
  globalThis.crypto = require('crypto');
}
if (!crypto.randomUUID) {
  crypto.randomUUID = function() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }
}

createApp(App).mount('#app')
