# 知识库项目说明（AI 工作规则）

## 项目目标

本项目是一个以 Markdown 为唯一知识内容来源的个人知识库，只有两个用途：

1. 用户通过网站阅读知识文章。
2. AI 工具直接读取 GitHub 仓库中的 Markdown，检索知识并回答问题。

网站只负责展示，不承担在线编辑、用户管理或内容存储职责。

## 目录说明

- `skills/`：存放项目规则、文章模板和任务指引。执行任务时按需阅读，不要无差别加载全部文件。
- `website/`：VitePress 网站源代码及知识文章。
- `website/articles/`：所有文章统一放在这里，不再按领域建立分类目录。
- `website/.vitepress/`：网站配置、主题和样式。
- `.github/workflows/`：GitHub Actions 构建与发布流程。

## 任务指引

开始任务前先判断任务类型，再阅读对应文件：

- 新建文章：阅读 `skills/article-writing.md`、`skills/article-template.md`、`skills/create-article.md`。
- 修改文章：阅读 `skills/article-writing.md`、`skills/update-article.md`。
- 修改网站：阅读 `skills/website-design.md`、`skills/modify-website.md`。
- 同时涉及文章和网站：分别读取两类指引，不要遗漏。

## 内容管理规则

1. 所有知识文章使用 Markdown。
2. 一篇文章对应一个独立文件夹，正文统一命名为 `index.md`。
3. 文章图片和附件放在该文章目录下的 `assets/` 中；没有资源时不必创建空目录。
4. 所有文章文件夹直接放在 `website/articles/` 下，不创建领域分类子目录。
5. 新建或删除文章时，同步更新 `website/articles/index.md` 和 VitePress 侧边栏配置。
6. 文章之间使用站内相对链接；图片使用相对路径。
7. 修改已有文章时保留原有技术含义、公式、代码和图表。用户没有要求重写时，不擅自改写正文。
8. 不重复创建已有主题的文章。先查看 `website/articles/index.md`，必要时搜索文章正文。
9. Markdown 源文件是知识内容的唯一可信来源；不要把构建生成的 HTML 当作原始知识。

## 网站维护规则

1. 网站使用 VitePress，依赖和命令统一由根目录 `package.json` 管理。
2. 网站只提供文章浏览、目录导航和全文搜索，不增加个人介绍、友链、评论、登录、在线编辑或数据库功能，除非用户明确提出。
3. 页面布局和视觉调整遵循 `skills/website-design.md`。
4. 修改文章显示样式时优先调整 `website/.vitepress/theme/custom.css`，不要为了小改动引入大型 UI 框架。
5. 保持桌面端阅读体验，同时确保窄屏下导航、代码块、表格和公式可用。
6. GitHub Pages 项目站点的 `base` 当前配置为 `/knowledge-base/`。如果仓库改名或改为自定义域名，必须同步检查该配置。

## 完成任务前的检查

修改完成后尽可能执行：

1. 检查 Markdown 代码围栏是否成对、图片路径和站内链接是否有效。
2. 运行 `npm run build`。
3. 如果构建失败，先修复再提交；如果因网络或环境限制无法安装依赖，必须明确说明没有完成本地构建验证。
4. 汇报实际修改的文件、验证结果和仍需用户操作的事项，不要声称未执行的检查已经通过。
