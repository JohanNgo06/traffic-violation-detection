import { useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  Clapperboard,
  LayoutDashboard,
  LogOut,
  Menu,
  Upload,
  X,
} from 'lucide-react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { clearAuth, getStoredUser } from '../utils/authStorage.js'

const navigation = [
  { label: 'Tổng quan', path: '/dashboard', icon: LayoutDashboard },
  { label: 'Tải video & Kẻ vạch', path: '/upload', icon: Upload },
  { label: 'Lịch sử vi phạm', path: '/violations', icon: AlertTriangle },
  { label: 'Video của tôi', path: '/sessions', icon: Clapperboard },
]

function MainLayout() {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [isMobileOpen, setIsMobileOpen] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const user = getStoredUser()
  const currentPage = navigation.find(({ path }) => path === location.pathname)

  function handleLogout() {
    clearAuth()
    navigate('/login', { replace: true })
  }

  return (
    <div className="min-h-screen bg-[#080e19] text-slate-100">
      {isMobileOpen && (
        <button
          aria-label="Đóng menu"
          className="fixed inset-0 z-30 bg-black/65 lg:hidden"
          onClick={() => setIsMobileOpen(false)}
          type="button"
        />
      )}

      <aside className={`fixed inset-y-0 left-0 z-40 flex w-[260px] flex-col border-r border-white/[0.06] bg-[#0b1423] transition-all duration-200 lg:translate-x-0 ${isMobileOpen ? 'translate-x-0' : '-translate-x-full'} ${isCollapsed ? 'lg:w-[76px]' : 'lg:w-[260px]'}`}>
        <div className={`flex h-[76px] items-center border-b border-white/[0.06] ${isCollapsed ? 'justify-center px-3' : 'justify-between px-5'}`}>
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-blue-400/20 bg-blue-500/10 text-blue-300">
              <Activity size={20} />
            </span>
            {!isCollapsed && (
              <div className="min-w-0">
                <p className="truncate text-xs font-bold tracking-wide text-white">TRAFFIC VISION</p>
                <p className="mt-1 truncate text-[9px] font-medium tracking-[0.18em] text-slate-500">CONTROL CENTER</p>
              </div>
            )}
          </div>
          <button
            aria-label="Đóng menu"
            className="rounded-lg p-2 text-slate-400 hover:bg-white/[0.06] lg:hidden"
            onClick={() => setIsMobileOpen(false)}
            type="button"
          >
            <X size={18} />
          </button>
        </div>

        <div className={`px-4 pb-2 pt-7 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-600 ${isCollapsed ? 'text-center' : ''}`}>
          {isCollapsed ? 'MENU' : 'Không gian làm việc'}
        </div>
        <nav className="flex-1 space-y-1 px-3">
          {navigation.map(({ icon: Icon, label, path }) => (
            <NavLink
              aria-label={isCollapsed ? label : undefined}
              className={({ isActive }) => `group flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium transition ${isActive ? 'bg-blue-500/[0.13] text-blue-200 ring-1 ring-inset ring-blue-400/10' : 'text-slate-400 hover:bg-white/[0.045] hover:text-slate-200'} ${isCollapsed ? 'justify-center lg:justify-center' : ''}`}
              key={path}
              onClick={() => setIsMobileOpen(false)}
              title={isCollapsed ? label : undefined}
              to={path}
            >
              <Icon className="shrink-0" size={18} />
              {(!isCollapsed || isMobileOpen) && <span className="truncate">{label}</span>}
              {(!isCollapsed || isMobileOpen) && path === '/violations' && location.pathname === path && (
                <span className="ml-auto h-1.5 w-1.5 rounded-full bg-blue-400" />
              )}
            </NavLink>
          ))}
        </nav>

        <div className={`border-t border-white/[0.06] p-3 ${isCollapsed ? 'hidden' : ''}`}>
          <div className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-3">
            <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              Hệ thống sẵn sàng
            </div>
            <p className="mt-2 text-[11px] leading-5 text-slate-500">Quản lý dữ liệu video và theo dõi hoạt động từ một nơi.</p>
          </div>
        </div>
        <button
          aria-label={isCollapsed ? 'Mở rộng sidebar' : 'Thu gọn sidebar'}
          className="hidden h-12 items-center justify-center border-t border-white/[0.06] text-slate-500 transition hover:bg-white/[0.04] hover:text-slate-300 lg:flex"
          onClick={() => setIsCollapsed((collapsed) => !collapsed)}
          type="button"
        >
          {isCollapsed ? <ChevronRight size={17} /> : <span className="flex items-center gap-2 text-xs">Thu gọn <ChevronLeft size={15} /></span>}
        </button>
      </aside>

      <div className={`min-h-screen transition-[padding] duration-200 ${isCollapsed ? 'lg:pl-[76px]' : 'lg:pl-[260px]'}`}>
        <header className="sticky top-0 z-20 flex h-[76px] items-center justify-between border-b border-white/[0.06] bg-[#080e19]/90 px-4 backdrop-blur-xl sm:px-7 lg:px-9">
          <div className="flex items-center gap-3">
            <button
              aria-label="Mở menu"
              className="rounded-lg p-2 text-slate-300 transition hover:bg-white/[0.06] lg:hidden"
              onClick={() => setIsMobileOpen(true)}
              type="button"
            >
              <Menu size={20} />
            </button>
            <div>
              <p className="text-xs text-slate-500">Không gian làm việc</p>
              <h1 className="mt-0.5 text-sm font-semibold text-slate-100">{currentPage?.label || 'Traffic Vision'}</h1>
            </div>
          </div>

          <div className="flex items-center gap-3 sm:gap-5">
            <div className="hidden text-right sm:block">
              <p className="max-w-48 truncate text-sm font-medium text-slate-200">{user?.fullName || 'Người dùng'}</p>
              <p className="mt-0.5 text-[11px] text-slate-500">Quản trị viên</p>
            </div>
            <div className="flex h-9 w-9 items-center justify-center rounded-full border border-blue-300/20 bg-blue-400/10 text-xs font-semibold text-blue-200">
              {(user?.fullName || 'U').trim().charAt(0).toUpperCase()}
            </div>
            <button
              aria-label="Đăng xuất"
              className="inline-flex items-center gap-2 rounded-lg border border-white/[0.08] px-3 py-2 text-xs font-medium text-slate-300 transition hover:border-rose-400/20 hover:bg-rose-400/[0.06] hover:text-rose-200 sm:text-sm"
              onClick={handleLogout}
              title="Đăng xuất"
              type="button"
            >
              <LogOut size={16} />
              <span className="hidden sm:inline">Đăng xuất</span>
              <ArrowUpRight className="hidden sm:block" size={13} />
            </button>
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1600px] p-4 sm:p-7 lg:p-9">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default MainLayout
