import i18next from 'i18next'
import { initReactI18next } from 'react-i18next'

const resources = {
  ru: {
    translation: {
      'nav.dashboard': 'Обзор',
      'nav.designer': 'Дизайнер экспериментов',
      'nav.monitor': 'Мониторинг',
      'nav.environments': 'Среды',
      'nav.arena': 'AlphaZero Arena',
      'nav.models': 'Модели',
      'nav.settings': 'Настройки',
    },
  },
  en: {
    translation: {
      'nav.dashboard': 'Dashboard',
      'nav.designer': 'Experiment Designer',
      'nav.monitor': 'Training Monitor',
      'nav.environments': 'Environments',
      'nav.arena': 'AlphaZero Arena',
      'nav.models': 'Model Zoo',
      'nav.settings': 'Settings',
    },
  },
}

i18next.use(initReactI18next).init({
  resources,
  lng: 'ru',
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
})

export default i18next
