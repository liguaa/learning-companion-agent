# 外部 Skill 扩展接口

当前只保留接入接口，不提供具体教学 Skill。默认启动不注册 Skill 工具，也不会扫描目录、下载 Skill 或加载外部模块。

## 接口约定

`learning_companion.skills.SkillProvider` 定义两个同步方法：

```python
def list_skills(self) -> Sequence[SkillInfo]: ...
def load_skill(self, name: str) -> str: ...
```

`SkillInfo` 包含名称和适用场景描述。名称使用 1～64 个字母、数字、连字符或下划线；描述最多 200 个字符；一次提供最多 20 个可用项。`load_skill` 返回 1～6000 个字符的文本指导。

外部适配器可以在内部对接文件目录或服务，访问方式和资源范围由适配器及宿主应用决定。本次没有实现任何目录或远程服务适配器。

## 接入方式

实现协议后，在应用代码中显式注入：

```python
from learning_companion.agent import open_companion
from learning_companion.config import load_settings

# external_provider 是你的外部适配器实例，实现 SkillProvider 的两个方法。
with open_companion(load_settings(), skill_provider=external_provider) as companion:
    session_id = companion.store.create_session()
    answer = companion.ask(session_id, "学习者的问题")
```

注册后，Agent 获得 `list_skills` 和 `load_skill` 两个 LangChain 工具；先查询目录，再按需加载指导。名称必须存在于当前目录，未注册名称不会传递给提供方的加载方法。

工具只交换元数据和文本指导，不执行 Skill 附带的脚本，不自动注册其他工具。工具权限继续由应用控制，Skill 正文不能覆盖全局教学规则。

目前没有 CLI 动态插件加载、自动安装或热加载入口。未来通过实现适配器并在应用启动时注入即可扩展，不需要修改 Agent 对话循环。
