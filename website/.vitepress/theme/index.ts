import DefaultTheme from 'vitepress/theme'
import { onMounted, nextTick, watch } from 'vue'
import { useRoute } from 'vitepress'
import './custom.css'

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()
    let mermaid
    const renderDiagrams = async () => {
      await nextTick()
      if (!mermaid) {
        const module = await import('mermaid')
        mermaid = module.default
        mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'default' })
      }
      const nodes = document.querySelectorAll('.mermaid:not([data-processed])')
      if (nodes.length) await mermaid.run({ nodes: Array.from(nodes) })
    }
    onMounted(renderDiagrams)
    watch(() => route.path, renderDiagrams)
  }
}
