import { useState } from 'react'
import { ArrowRight, LoaderCircle, LockKeyhole, Mail, UserRound } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import axiosClient from '../api/axiosClient.js'
import AuthPageShell from '../components/AuthPageShell.jsx'

function RegisterPage() {
  const navigate = useNavigate()
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  async function handleSubmit(event) {
    event.preventDefault()
    setErrorMessage('')

    const normalizedName = fullName.trim()
    const normalizedEmail = email.trim()
    if (!normalizedName) {
      setErrorMessage('Vui lòng nhập họ và tên.')
      return
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail)) {
      setErrorMessage('Vui lòng nhập địa chỉ email hợp lệ.')
      return
    }
    if (password.length < 8) {
      setErrorMessage('Mật khẩu cần có ít nhất 8 ký tự.')
      return
    }
    if (password !== confirmPassword) {
      setErrorMessage('Mật khẩu xác nhận không khớp.')
      return
    }

    setIsLoading(true)
    try {
      const { data } = await axiosClient.post('/auth/register', {
        fullName: normalizedName,
        email: normalizedEmail,
        password,
      })

      navigate('/login', {
        replace: true,
        state: {
          successMessage: data?.message || 'Đăng ký thành công. Vui lòng đăng nhập.',
        },
      })
    } catch (error) {
      setErrorMessage(error.message || 'Đăng ký thất bại. Vui lòng thử lại.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <AuthPageShell
      mode="register"
      title="Tạo tài khoản"
      subtitle="Đăng ký để quản lý video và theo dõi các vi phạm giao thông."
    >
      {errorMessage && (
        <div className="mb-5 rounded-xl border border-rose-400/20 bg-rose-400/[0.07] px-4 py-3 text-sm text-rose-200" role="alert">
          {errorMessage}
        </div>
      )}

      <form className="space-y-4" onSubmit={handleSubmit}>
        <InputField
          autoComplete="name"
          icon={<UserRound size={17} />}
          id="fullName"
          label="Họ và tên"
          onChange={setFullName}
          placeholder="Nguyễn Văn A"
          value={fullName}
        />
        <InputField
          autoComplete="email"
          icon={<Mail size={17} />}
          id="email"
          label="Email"
          onChange={setEmail}
          placeholder="ten@congty.vn"
          type="email"
          value={email}
        />
        <InputField
          autoComplete="new-password"
          icon={<LockKeyhole size={17} />}
          id="password"
          label="Mật khẩu"
          onChange={setPassword}
          placeholder="Tối thiểu 8 ký tự"
          type="password"
          value={password}
        />
        <InputField
          autoComplete="new-password"
          icon={<LockKeyhole size={17} />}
          id="confirmPassword"
          label="Xác nhận mật khẩu"
          onChange={setConfirmPassword}
          placeholder="Nhập lại mật khẩu"
          type="password"
          value={confirmPassword}
        />
        <button
          className="mt-2 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-blue-500 px-4 py-3 text-sm font-semibold text-white transition hover:bg-blue-400 focus:outline-none focus:ring-4 focus:ring-blue-400/20 disabled:cursor-not-allowed disabled:opacity-65"
          disabled={isLoading}
          type="submit"
        >
          {isLoading ? <LoaderCircle className="animate-spin" size={17} /> : null}
          {isLoading ? 'Đang tạo tài khoản...' : 'Tạo tài khoản'}
          {!isLoading && <ArrowRight size={17} />}
        </button>
      </form>

      <p className="mt-6 text-center text-sm text-slate-400">
        Đã có tài khoản?{' '}
        <Link className="font-semibold text-blue-300 transition hover:text-blue-200" to="/login">
          Đăng nhập
        </Link>
      </p>
    </AuthPageShell>
  )
}

function InputField({ autoComplete, icon, id, label, onChange, placeholder, type = 'text', value }) {
  return (
    <div>
      <label className="mb-2 block text-sm font-medium text-slate-300" htmlFor={id}>{label}</label>
      <div className="relative">
        <span className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500">{icon}</span>
        <input
          autoComplete={autoComplete}
          className="w-full rounded-xl border border-slate-700/80 bg-[#0d1726] py-3 pl-11 pr-4 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-blue-400/70 focus:ring-4 focus:ring-blue-500/10"
          id={id}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          required
          type={type}
          value={value}
        />
      </div>
    </div>
  )
}

export default RegisterPage
