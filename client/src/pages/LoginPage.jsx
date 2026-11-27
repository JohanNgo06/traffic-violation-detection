import { useState } from 'react'
import { ArrowRight, LoaderCircle, LockKeyhole, Mail } from 'lucide-react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import axiosClient from '../api/axiosClient.js'
import AuthPageShell from '../components/AuthPageShell.jsx'
import { saveAuth } from '../utils/authStorage.js'

function LoginPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const successMessage = location.state?.successMessage

  async function handleSubmit(event) {
    event.preventDefault()
    setErrorMessage('')

    const normalizedEmail = email.trim()
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail)) {
      setErrorMessage('Vui lòng nhập địa chỉ email hợp lệ.')
      return
    }

    if (!password) {
      setErrorMessage('Vui lòng nhập mật khẩu.')
      return
    }

    setIsLoading(true)
    try {
      const { data } = await axiosClient.post('/auth/login', {
        email: normalizedEmail,
        password,
      })

      if (!data?.token || !data?.user) {
        throw new Error('Dữ liệu đăng nhập trả về không hợp lệ. Vui lòng liên hệ quản trị viên.')
      }

      saveAuth(data.token, data.user)
      navigate('/dashboard', { replace: true })
    } catch (error) {
      setErrorMessage(error.message || 'Đăng nhập thất bại. Vui lòng thử lại.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <AuthPageShell
      mode="login"
      title="Đăng nhập"
      subtitle="Nhập thông tin tài khoản để truy cập không gian quản trị."
    >
      {successMessage && (
        <div className="mb-5 rounded-xl border border-emerald-400/20 bg-emerald-400/[0.07] px-4 py-3 text-sm text-emerald-200" role="status">
          {successMessage}
        </div>
      )}
      {errorMessage && <FormAlert message={errorMessage} />}

      <form className="space-y-5" onSubmit={handleSubmit}>
        <div>
          <label className="mb-2 block text-sm font-medium text-slate-300" htmlFor="email">Email</label>
          <div className="relative">
            <Mail className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" size={17} />
            <input
              autoComplete="email"
              className="w-full rounded-xl border border-slate-700/80 bg-[#0d1726] py-3 pl-11 pr-4 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-blue-400/70 focus:ring-4 focus:ring-blue-500/10"
              id="email"
              onChange={(event) => setEmail(event.target.value)}
              placeholder="ten@congty.vn"
              required
              type="email"
              value={email}
            />
          </div>
        </div>
        <div>
          <label className="mb-2 block text-sm font-medium text-slate-300" htmlFor="password">Mật khẩu</label>
          <div className="relative">
            <LockKeyhole className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" size={17} />
            <input
              autoComplete="current-password"
              className="w-full rounded-xl border border-slate-700/80 bg-[#0d1726] py-3 pl-11 pr-4 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-blue-400/70 focus:ring-4 focus:ring-blue-500/10"
              id="password"
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Nhập mật khẩu"
              required
              type="password"
              value={password}
            />
          </div>
        </div>
        <button
          className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-blue-500 px-4 py-3 text-sm font-semibold text-white transition hover:bg-blue-400 focus:outline-none focus:ring-4 focus:ring-blue-400/20 disabled:cursor-not-allowed disabled:opacity-65"
          disabled={isLoading}
          type="submit"
        >
          {isLoading ? <LoaderCircle className="animate-spin" size={17} /> : null}
          {isLoading ? 'Đang đăng nhập...' : 'Đăng nhập'}
          {!isLoading && <ArrowRight size={17} />}
        </button>
      </form>

      <p className="mt-7 text-center text-sm text-slate-400">
        Chưa có tài khoản?{' '}
        <Link className="font-semibold text-blue-300 transition hover:text-blue-200" to="/register">
          Tạo tài khoản
        </Link>
      </p>
    </AuthPageShell>
  )
}

function FormAlert({ message }) {
  return (
    <div className="mb-5 rounded-xl border border-rose-400/20 bg-rose-400/[0.07] px-4 py-3 text-sm text-rose-200" role="alert">
      {message}
    </div>
  )
}

export default LoginPage
