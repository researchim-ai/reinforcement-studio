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
      'nav.sweeps': 'Sweeps',
      'nav.plugins': 'Свои плагины',
      'nav.networkBuilder': 'Архитектуры сетей',
      'nav.sceneBuilder': 'Конструктор сред',
      'nav.academy': 'Учебный центр',
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
      'nav.sweeps': 'Sweeps',
      'nav.plugins': 'Custom Plugins',
      'nav.networkBuilder': 'Network Builder',
      'nav.sceneBuilder': 'Scene Builder',
      'nav.academy': 'Learning Center',
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
