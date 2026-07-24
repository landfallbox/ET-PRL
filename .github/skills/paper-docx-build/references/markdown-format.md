# Markdown Input Conventions

## General Conversion

Use standard Pandoc Markdown. Relative image paths are resolved relative to the input Markdown file.

Inline and display mathematics should use TeX dollar delimiters, for example `$x_t$` and `$$y = f(x)$$`.

## ET-PRL Algorithm Table Injection

The bundled native-Word algorithm table is specific to the ET-PRL online control procedure. To replace its Markdown placeholder, use the exact title marker supplied to `--algorithm-title`, then put the placeholder pseudocode in one fenced code block before the next heading.

````markdown
**Algorithm 1. Event-triggered predictive reinforcement learning with unsupervised dynamic event gating (ET-PRL) online control procedure**

```text
Input: ...
Output: ...
...
```
````

The converter removes the title and consecutive code-block paragraphs, then inserts the bundled table containing native Word OMML equations. Do not use this option for unrelated algorithms; omit `--algorithm-title` to preserve the Markdown content as converted by Pandoc.