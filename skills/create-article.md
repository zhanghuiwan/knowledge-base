# 新建文章指引

执行新建文章任务时按以下步骤进行：

1. 阅读 `AGENTS.md` 和 `skills/article-writing.md`、`skills/article-template.md`。
2. 查看 `website/articles/index.md`，必要时搜索现有文章，确认主题是否已存在。
3. 确定文章范围和稳定的英文目录名，不按领域创建额外目录。
4. 在 `website/articles/<文章目录>/` 中创建 `index.md`；只有文章确实包含图片或附件时才创建 `assets/`。
5. 按文章编写规范添加 Frontmatter、标题、正文、公式、代码、图表和参考资料。
6. 图片放在本篇文章的 `assets/` 中，使用相对路径引用。
7. 更新 `website/articles/index.md`，并在 `website/.vitepress/config.mts` 中增加侧边栏入口。
8. 检查重复内容、标题层级、链接、图片路径和公式/代码围栏。
9. 运行 `npm run build`。如果环境无法安装依赖或执行构建，明确说明原因。
10. 最终报告新增文件、目录入口和检查结果。
