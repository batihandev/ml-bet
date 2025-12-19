// @ts-check
import withNuxt from './.nuxt/eslint.config.mjs'
import eslintConfigPrettier from 'eslint-config-prettier'

export default withNuxt(
  // 1. Disable all formatting-related ESLint rules (indent, quotes, semi, etc.)
  ...eslintConfigPrettier,

  // 2. Your project-specific rules can go here if you want
  {
    rules: {
      // e.g. turn off anything still bothering you, or add real quality rules
      // "no-console": "warn",
    }
  },
)
