# 修改网站指引

1. 先阅读 `AGENTS.md` 和 `skills/website-design.md`，再检查当前 VitePress 配置与主题实现。
2. 明确本次修改属于布局、视觉样式、导航、搜索、文章渲染还是响应式适配。
3. 尽量沿用现有 VitePress 组件和配置，不重建已有功能。
4. 视觉样式优先修改 `website/.vitepress/theme/custom.css`；站点功能和导航修改 `website/.vitepress/config.mts`；只有必要时才新增组件。
5. 不要把文章正文复制到 Vue/TS 文件中；Markdown 仍然是内容源。
6. 不要增加数据库、服务端接口、登录、评论或在线编辑能力，除非用户明确要求。
7. 检查首页、文章列表、两篇现有文章、搜索、目录、深色模式和移动端。
8. 运行 `npm run build`；无法构建时说明环境限制，不要声称已通过。
9. 最终报告修改内容、验证情况和需要用户确认的视觉取舍。
