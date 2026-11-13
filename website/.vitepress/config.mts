import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  lang: 'zh-CN',
  title: '技术知识库',
  description: '以 Markdown 为源的技术知识库，支持网页阅读与 AI 检索。',
  base: '/knowledge-base/',
  lastUpdated: true,
  cleanUrls: true,

  markdown: {
    math: true,
    lineNumbers: true,
    image: {
      lazyLoading: true,
    },
  },

  mermaid: {
    theme: 'default',
    securityLevel: 'strict',
  },
  mermaidPlugin: {
    class: 'mermaid knowledge-diagram',
  },

  themeConfig: {
    siteTitle: '知识库',
    nav: [
      { text: '首页', link: '/' },
      { text: '全部文章', link: '/articles/' },
    ],

    sidebar: {
      '/articles/': [
        {
          text: '文章',
          items: [
            { text: 'RT-DETR', link: '/articles/rt-detr/' },
            { text: 'RF-DETR', link: '/articles/rf-detr/' },
          ],
        },
      ],
    },

    outline: {
      level: [2, 3],
      label: '本页目录',
    },

    search: {
      provider: 'local',
      options: {
        detailedView: true,
      },
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },

    lastUpdated: {
      text: '最后更新于',
    },

    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '文章导航',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换浅色模式',
    darkModeSwitchTitle: '切换深色模式',
    skipToContentLabel: '跳转到正文',
  },
}))
