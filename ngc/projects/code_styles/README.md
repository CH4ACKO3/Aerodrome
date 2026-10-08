# 零号工程 00.13：三种代码写法

同一个一维速度闭环分别使用脚本式、函数式和面向对象式组织，三种写法共用相同动力学并产生相同轨迹。

完整课程、数学说明、命令、练习与状态生命周期见[代码编写教学](../../docs/code-styles.md)。组件索引见[零号工程 00.13](../../docs/project-zero.md#coding-styles)。

在 `ngc/` 中运行：

```sh
uv run --locked python projects/code_styles/script_style.py
uv run --locked python projects/code_styles/functional_style.py
uv run --locked python projects/code_styles/object_style.py
```

三个程序均支持 `--output PATH`，默认分别写到 `artifacts/styles/script`、`functional`、`object`。
