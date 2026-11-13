import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({
  base: '/knowledge-base/',
  lang: 'zh-CN',
  title: '技术知识库',
  description: '以 Markdown 为核心的个人技术知识库',
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    lineNumbers: true,
    config(md) { md.use(mathjax3) }
  },
  themeConfig: {
    nav: [{ text: '首页', link: '/' }, { text: '文章', link: '/articles/' }],
    sidebar: { '/articles/': [{ text: '知识文章', items: [
      { text: '文章总览', link: '/articles/' },
      { text: 'RT-DETR', link: '/articles/rt-detr/' },
      { text: 'RF-DETR', link: '/articles/rf-detr/' }
    ]}]},
    outline: { level: [2, 3], label: '本页目录' },
    search: { provider: 'local' },
    socialLinks: [{ icon: 'github', link: 'https://github.com/zhanghuiwan/knowledge-base' }],
    footer: { message: '内容以 Markdown 源文件为准', copyright: 'Powered by VitePress' }
  }
})
