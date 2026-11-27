import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  Clapperboard,
  Clock3,
  Upload,
} from 'lucide-react'

const pageDetails = {
  dashboard: {
    eyebrow: 'Tổng quan hệ thống',
    title: 'Trung tâm điều hành',
    description: 'Theo dõi nhanh video đã xử lý và các sự kiện vi phạm trong hệ thống.',
    icon: Activity,
    items: ['Tổng quan hoạt động', 'Trạng thái xử lý video', 'Vi phạm gần đây'],
  },
  upload: {
    eyebrow: 'Quản lý dữ liệu',
    title: 'Tải video & Kẻ vạch',
    description: 'Tải video lên và thiết lập vùng quan tâm trước khi bắt đầu phân tích.',
    icon: Upload,
    items: ['Tải video lên hệ thống', 'Cấu hình vạch dừng', 'Thiết lập làn đường'],
  },
  violations: {
    eyebrow: 'Phân tích sự kiện',
    title: 'Lịch sử vi phạm',
    description: 'Tra cứu các sự kiện được ghi nhận theo video, phương tiện và loại vi phạm.',
    icon: AlertTriangle,
    items: ['Lọc theo video', 'Lọc theo phương tiện', 'Xem bằng chứng vi phạm'],
  },
  sessions: {
    eyebrow: 'Thư viện video',
    title: 'Video của tôi',
    description: 'Quản lý video đã tải lên và theo dõi trạng thái các phiên phân tích.',
    icon: Clapperboard,
    items: ['Danh sách phiên video', 'Trạng thái xử lý', 'Cấu hình từng phiên'],
  },
}

export function DashboardPage() {
  return <WorkspacePage page="dashboard" />
}

export function UploadPage() {
  return <WorkspacePage page="upload" />
}

export function ViolationsPage() {
  return <WorkspacePage page="violations" />
}

export function SessionsPage() {
  return <WorkspacePage page="sessions" />
}

function WorkspacePage({ page }) {
  const details = pageDetails[page]
  const Icon = details.icon

  return (
    <div className="space-y-7">
      <section className="overflow-hidden rounded-2xl border border-white/[0.07] bg-gradient-to-br from-[#101d31] via-[#0d1727] to-[#0b1422] p-6 sm:p-8">
        <div className="flex flex-col justify-between gap-6 sm:flex-row sm:items-end">
          <div className="max-w-2xl">
            <div className="mb-4 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-blue-300">
              <span className="h-px w-5 bg-blue-400/70" />
              {details.eyebrow}
            </div>
            <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">{details.title}</h2>
            <p className="mt-3 text-sm leading-6 text-slate-400 sm:text-base">{details.description}</p>
          </div>
          <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border border-blue-300/15 bg-blue-400/[0.08] text-blue-200">
            <Icon size={25} />
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        {details.items.map((item, index) => (
          <div className="rounded-2xl border border-white/[0.07] bg-[#0c1523] p-5 transition hover:border-blue-300/15" key={item}>
            <div className="flex items-start justify-between">
              <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/[0.04] text-slate-300">
                {index === 0 ? <Icon size={18} /> : index === 1 ? <Clock3 size={18} /> : <ArrowUpRight size={18} />}
              </span>
              <span className="rounded-full border border-amber-300/15 bg-amber-300/[0.06] px-2.5 py-1 text-[10px] font-medium text-amber-200">
                Đang phát triển
              </span>
            </div>
            <h3 className="mt-5 text-sm font-semibold text-slate-200">{item}</h3>
            <p className="mt-2 text-xs leading-5 text-slate-500">
              Tính năng sẽ hiển thị dữ liệu thực sau khi được kết nối với API tương ứng.
            </p>
          </div>
        ))}
      </section>

      <section className="flex flex-col gap-3 rounded-2xl border border-dashed border-slate-700/70 bg-white/[0.015] px-5 py-6 sm:flex-row sm:items-center sm:gap-4 sm:px-6">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-700/70 bg-slate-800/50 text-slate-400">
          <Activity size={18} />
        </span>
        <div>
          <h3 className="text-sm font-medium text-slate-300">Chưa có dữ liệu để hiển thị</h3>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            Đây là trang giao diện khung. Dữ liệu và thao tác sẽ được tích hợp khi các luồng API tương ứng sẵn sàng.
          </p>
        </div>
      </section>
    </div>
  )
}
