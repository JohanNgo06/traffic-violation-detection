export function saveAuth(token, user) {
  localStorage.setItem('token', token)
  localStorage.setItem('user', JSON.stringify(user))
}

export function getStoredUser() {
  const rawUser = localStorage.getItem('user')

  if (!rawUser) {
    return null
  }

  try {
    return JSON.parse(rawUser)
  } catch {
    return null
  }
}

export function clearAuth() {
  localStorage.removeItem('token')
  localStorage.removeItem('user')
}
