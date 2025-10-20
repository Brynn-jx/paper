import os

def main():
    print("=== 📄 自动化论文总结模板生成器 ===")

    title = input("论文标题: ").strip()
    author = input("作者: ").strip()
    conference = input("会议/期刊: ").strip()
    background = input("研究背景: ").strip()
    method = input("方法概述: ").strip()
    innovation = input("创新点: ").strip()
    dataset = input("数据集: ").strip()
    results = input("主要结果: ").strip()
    pros = input("优点: ").strip()
    cons = input("缺点: ").strip()
    insight = input("个人启发: ").strip()

    # 指定保存路径
    save_dir = r"D:\笔记\每周汇报"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{title.replace(' ', '_')}.md"
    file_path = os.path.join(save_dir, filename)

    content = f"""# {title}
**作者**: {author}  
**会议/期刊**: {conference}  

---

## 1. 研究背景
{background}

## 2. 方法概述
{method}

## 3. 创新点
{innovation}

## 4. 实验与结果
- 数据集: {dataset}
- 主要结果: {results}

## 5. 优缺点分析
- 优点: {pros}
- 缺点: {cons}

## 6. 个人启发
{insight}
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n✅ 论文总结已生成: {file_path}")

if __name__ == "__main__":
    main()
