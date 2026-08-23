import { create } from 'zustand'

interface SettingsState {
  language: 'ru' | 'en'
  toggleLanguage: () => void
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  language: 'ru',
  toggleLanguage: () => set({ language: get().language === 'ru' ? 'en' : 'ru' }),
}))
