import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/knowledge-base/',
  lang: 'zh-CN',
  title: '技术知识库',
  description: '以 Markdown 为核心的个人技术知识库',
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    math: true,
    lineNumbers: true,
    config(md) {
      const defaultFence = md.renderer.rules.fence
      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        if (token.info.trim() === 'mermaid') {
          return '<pre class="mermaid">' + md.utils.escapeHtml(token.content) + '</pre>'
        }
        return defaultFence(tokens, idx, options, env, self)
      }
    }
  },
  themeConfig: {
    nav: [{ text: '首页', link: '/' }, { text: '文章', link: '/articles/' }],
    sidebar: { '/articles/': [{ text: '知识文章', items: [
      { text: '文章总览', link: '/articles/' },
      { text: 'YOLO11 / YOLO26', link: '/articles/yolo11-yolo26/' },
      { text: 'DETR', link: '/articles/detr/' },
      { text: 'RT-DETR', link: '/articles/rt-detr/' },
      { text: 'RF-DETR', link: '/articles/rf-detr/' },
      { text: 'DINOv3', link: '/articles/dinov3/' },
      { text: 'ZGCM-1', link: '/articles/zgcm-1/' },
      { text: 'DEIMv2', link: '/articles/deimv2/' }
    ]}]},
    outline: { level: [2, 3], label: '本页目录' },
    search: { provider: 'local' },
    socialLinks: [{ icon: 'github', link: 'https://github.com/zhanghuiwan/knowledge-base' }],
    footer: { message: '内容以 Markdown 源文件为准', copyright: 'Powered by VitePress' }
  }
})
