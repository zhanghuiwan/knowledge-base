# 技术知识库

一个以 Markdown 为内容源的轻量知识库。网站使用 VitePress 构建，通过 GitHub Pages 发布；AI 工具可以直接读取仓库中的 Markdown 原文。

## 目录结构

```text
knowledge-base/
├── AGENTS.md
├── README.md
├── package.json
├── skills/                 # 项目规则、文章模板和任务指引
├── .github/workflows/      # 自动构建与部署
└── website/                # VitePress 网站
    ├── .vitepress/         # 配置和主题
    ├── public/             # 公共静态资源与 llms.txt
    ├── index.md            # 网站首页
    └── articles/           # 所有知识文章
        └── <article>/
            ├── index.md
            └── assets/     # 该文章专属图片和附件（按需创建）
```

## 本地运行

需要 Node.js 20 或更高版本。

```bash
npm install
npm run dev
```

构建和预览：

```bash
npm run build
npm run preview
```

## 新建文章

1. 阅读 `AGENTS.md` 和 `skills/` 中与任务相关的指引。
2. 在 `website/articles/` 下新建一个英文短横线命名的文章目录。
3. 在目录中创建 `index.md`；文章图片放在同目录的 `assets/` 中。
4. 更新 `website/articles/index.md` 和 `website/.vitepress/config.mts` 中的侧边栏。
5. 执行 `npm run build` 检查。

## 发布

推送到 `main` 分支后，GitHub Actions 会自动安装依赖、构建 VitePress 并部署到 GitHub Pages。

首次发布前，需要在 GitHub 仓库的 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**。

当前仓库名预设为 `knowledge-base`，对应站点地址：

`https://zhanghuiwan.github.io/knowledge-base/`
