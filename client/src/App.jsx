
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import PrivateRoute from './components/PrivateRoute.jsx'
import MainLayout from './layouts/MainLayout.jsx'
import LoginPage from './pages/LoginPage.jsx'
import RegisterPage from './pages/RegisterPage.jsx'
import {
  DashboardPage,
  SessionsPage,
  UploadPage,
  ViolationsPage,
} from './pages/WorkspacePages.jsx'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route element={<PrivateRoute />}>
          <Route element={<MainLayout />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/violations" element={<ViolationsPage />} />
            <Route path="/sessions" element={<SessionsPage />} />
          </Route>
        </Route>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
