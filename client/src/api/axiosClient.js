import axios from 'axios'
import { clearAuth } from '../utils/authStorage.js'

const axiosClient = axios.create({
  baseURL: 'http://localhost:5000/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

axiosClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')

  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }

  return config
})

axiosClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status

    if (status === 401) {
      clearAuth()

      if (window.location.pathname !== '/login') {
        window.location.assign('/login')
      }
    }

    if (error.response) {
      const { message } = error.response.data ?? {}
      if (typeof message === 'string' && message.trim()) {
        error.message = message
      } else if (status >= 500) {
        error.message = 'Máy chủ đang gặp sự cố. Vui lòng thử lại sau.'
      } else if (!error.message) {
        error.message = 'Yêu cầu không thành công. Vui lòng kiểm tra lại.'
      }
    } else if (error.request) {
      error.message = 'Không thể kết nối máy chủ. Vui lòng kiểm tra kết nối mạng.'
    }

    return Promise.reject(error)
  },
)

export default axiosClient
