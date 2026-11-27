import { Activity, ArrowUpRight, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'

function AuthPageShell({ children, mode, subtitle, title }) {
  return (
    <main className="min-h-screen bg-[#070d18] text-slate-100">
      <div className="mx-auto grid min-h-screen max-w-[1440px] lg:grid-cols-[1.05fr_0.95fr]">
        <section className="relative hidden overflow-hidden border-r border-white/[0.06] bg-[#0b1423] px-12 py-10 lg:flex lg:flex-col lg:justify-between xl:px-20">
          <div className="pointer-events-none absolute -left-32 top-1/4 h-96 w-96 rounded-full bg-blue-500/[0.08] blur-3xl" />
          <div className="pointer-events-none absolute -right-36 bottom-0 h-96 w-96 rounded-full bg-cyan-400/[0.05] blur-3xl" />

          <Link className="relative inline-flex w-fit items-center gap-3" to="/">
            <span className="flex h-11 w-11 items-center justify-center rounded-xl border border-blue-400/20 bg-blue-500/10 text-blue-300">
              <Activity size={21} />
            </span>
            <span>
              <span className="block text-sm font-bold tracking-wide text-white">TRAFFIC VISION</span>
              <span className="mt-0.5 block text-[10px] font-medium tracking-[0.2em] text-slate-500">INTELLIGENCE PLATFORM</span>
            </span>
          </Link>

          <div className="relative max-w-xl py-14">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-blue-400/15 bg-blue-400/[0.06] px-3 py-1.5 text-xs font-medium text-blue-200">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              Giám sát giao thông thông minh
            </div>
            <h1 className="text-4xl font-semibold leading-[1.15] tracking-tight text-white xl:text-5xl">
              Quan sát rõ hơn.
              <br />
              <span className="text-blue-300">Vận hành an toàn hơn.</span>
            </h1>
            <p className="mt-6 max-w-md text-base leading-7 text-slate-400">
              Nền tảng phân tích video hỗ trợ phát hiện và quản lý vi phạm giao thông trong một không gian làm việc tập trung.
            </p>

            <div className="mt-10 grid max-w-md grid-cols-2 gap-3">
              <Feature icon={<ShieldCheck size={17} />} label="Theo dõi vi phạm" />
              <Feature icon={<Activity size={17} />} label="Phân tích video" />
            </div>
          </div>

          <div className="relative flex items-center justify-between border-t border-white/[0.07] pt-5 text-xs text-slate-500">
            <span>Traffic Violation Detection</span>
            <span className="inline-flex items-center gap-1.5">Hệ thống quản trị <ArrowUpRight size={13} /></span>
          </div>
        </section>

        <section className="flex min-h-screen flex-col px-5 py-7 sm:px-10 lg:px-14 xl:px-24">
          <Link className="mb-12 inline-flex items-center gap-2 self-start lg:hidden" to="/">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-blue-400/20 bg-blue-500/10 text-blue-300">
              <Activity size={18} />
            </span>
            <span className="text-xs font-bold tracking-wide text-white">TRAFFIC VISION</span>
          </Link>
          <div className="m-auto w-full max-w-[420px] py-6">
            <div className="mb-8">
              <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-blue-300">
                {mode === 'login' ? 'Chào mừng trở lại' : 'Bắt đầu sử dụng'}
              </p>
              <h2 className="text-3xl font-semibold tracking-tight text-white">{title}</h2>
              <p className="mt-2 text-sm leading-6 text-slate-400">{subtitle}</p>
            </div>
            {children}
          </div>
          <p className="pt-8 text-center text-xs text-slate-600">
            Bằng việc tiếp tục, bạn đồng ý sử dụng hệ thống theo chính sách của đơn vị.
          </p>
        </section>
      </div>
    </main>
  )
}

function Feature({ icon, label }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-white/[0.07] bg-white/[0.025] px-4 py-3 text-sm text-slate-300">
      <span className="text-blue-300">{icon}</span>
      {label}
    </div>
  )
}

export default AuthPageShell
