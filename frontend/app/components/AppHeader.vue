<template>
  <UHeader
    class="border-b border-gray-200/60 dark:border-gray-800/80 sticky top-0 bg-background/75 backdrop-blur-sm z-50"
  >
    <template #left>
      <NuxtLink to="/" class="flex items-center gap-2">
        <div
          class="flex h-8 w-8 items-center justify-center rounded-xl bg-emerald-500 text-xs font-semibold text-white"
        >
          FE
        </div>
        <span class="text-sm font-semibold tracking-tight">
          Football Edge Lab
        </span>
      </NuxtLink>
    </template>

    <template #right>
      <UColorModeButton />
    </template>
  </UHeader>
</template>
