# 修改文章指引

1. 先读取目标文章完整内容，确认修改范围。
2. 阅读 `skills/article-writing.md`，并识别用户要求的是修正、补充、重组还是重写。
3. 用户只要求局部修改时，不要顺手重写其他章节。
4. 保留原有技术含义、公式、代码、图表和引用；确需调整时说明原因。
5. 新增图片或附件时放入本篇文章的 `assets/`，并使用相对路径。
6. 更新 Frontmatter 中的 `updated` 日期。
7. 如果文章标题或目录发生变化，检查 `website/articles/index.md`、侧边栏和站内引用。
8. 检查 Markdown 结构、公式、Mermaid、图片和站内链接，最后运行 `npm run build`。
