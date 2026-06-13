import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {},
});

// Add a request interceptor to attach the JWT token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      if (!config.headers) {
        config.headers = {};
      }
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor for global error handling and token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // If the error is 401 and not already retried
    if (error.response && error.response.status === 401 && !originalRequest._retry) {
      if (originalRequest.url === '/api/auth/token' || originalRequest.url === '/api/auth/refresh') {
        // If login or refresh failed, just logout
        localStorage.removeItem('token');
        localStorage.removeItem('refresh_token');
        localStorage.removeItem('user');
        if (window.location.pathname !== '/login') {
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }

      originalRequest._retry = true;
      const refreshToken = localStorage.getItem('refresh_token');

      if (refreshToken) {
        try {
          const response = await authAPI.refreshToken(refreshToken);
          const { access_token } = response.data;
          
          localStorage.setItem('token', access_token);
          api.defaults.headers.Authorization = `Bearer ${access_token}`;
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          
          return api(originalRequest);
        } catch (refreshError) {
          // Refresh failed
          localStorage.removeItem('token');
          localStorage.removeItem('refresh_token');
          localStorage.removeItem('user');
          window.location.href = '/login';
          return Promise.reject(refreshError);
        }
      }
    }

    return Promise.reject(error);
  }
);

// Auth API calls
export const authAPI = {
  login: (username, password) => {
    const params = new URLSearchParams();
    params.append('username', username);
    params.append('password', password);
    return api.post('/api/auth/token', params, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });
  },
  register: (email, password, fullName = null) => api.post('/api/auth/register', { email, password, full_name: fullName }),
  refreshToken: (refreshToken) => api.post('/api/auth/refresh', { refresh_token: refreshToken }),
  logout: (refreshToken) => api.post('/api/auth/logout', { refresh_token: refreshToken }),
  getCurrentUser: () => api.get('/api/auth/me'),
  changePassword: (oldPassword, newPassword) => api.put('/api/auth/password', { old_password: oldPassword, new_password: newPassword }),
};

// Users API calls
export const usersAPI = {
  getProfile: () => api.get('/api/users/me'),
  updateProfile: (data) => api.put('/api/users/me', data),
  getDashboardStats: () => api.get('/api/users/me/dashboard-stats'),
  getSettings: () => api.get('/api/users/me/settings'),
  updateSettings: (data) => api.put('/api/users/me/settings', data),
};

// Resume API calls
export const resumeAPI = {
  getResumes: () => api.get('/api/resumes'),
  getResume: (id) => api.get(`/api/resumes/${id}`),
  createResume: (data) => api.post('/api/resumes', data),
  updateResume: (id, data) => api.put(`/api/resumes/${id}`, data),
  deleteResume: (id) => api.delete(`/api/resumes/${id}`),
  deleteResumes: (ids) => api.post('/api/resumes/bulk-delete', { ids }),
  uploadResume: (file, title, targetRole) => {
    const formData = new FormData();
    formData.append('file', file);
    if (title) formData.append('title', title);
    if (targetRole) formData.append('target_role', targetRole);
    // Debug: log FormData contents
    for (let pair of formData.entries()) {
      console.log('FormData:', pair[0], pair[1]);
    }
    // Do NOT set Content-Type manually; Axios will handle it for FormData
    return api.post('/api/resumes/upload', formData);
  },
  getResumeVersions: (id) => api.get(`/api/resumes/${id}/versions`),
  downloadResume: (id) => api.get(`/api/resumes/${id}/download`, { responseType: 'blob' }),
  getResumePreview: (id) => api.get(`/api/resumes/${id}/preview`),
  getPipelineStatus: (id) => api.get(`/api/resumes/${id}/pipeline-status`),
  processResumePipeline: (id) => api.post(`/api/resumes/${id}/process-pipeline`),
  atsCheck: (id, jobDescription) => api.post(`/api/resumes/${id}/ats-check`, { job_description: jobDescription }),
  optimizeResume: (id, optimizationType, jobDescription = "", jobId = null, saveAsNew = false) => 
    api.post(`/api/resumes/${id}/optimize`, { 
      optimization_type: optimizationType,
      job_description: jobDescription,
      job_id: jobId,
      save_as_new: saveAsNew
    }),
};

// Interview API calls
export const interviewAPI = {
  generateQuestions: (data) => api.post('/api/interview/questions', data),
  createSession: (data) => api.post('/api/interview/sessions', data),
  getSessions: (status, limit = 10) => {
    const url = status ? `/api/interview/sessions?status=${status}&limit=${limit}` : `/api/interview/sessions?limit=${limit}`;
    return api.get(url);
  },
  getSession: (id) => api.get(`/api/interview/sessions/${id}`),
  getSessionQuestions: (id) => api.get(`/api/interview/sessions/${id}/questions`),
  submitAnswer: (sessionId, data) => api.post(`/api/interview/sessions/${sessionId}/answer`, data),
  getFeedback: (sessionId, questionId) => api.post(`/api/interview/sessions/${sessionId}/feedback/${questionId}`),
  completeSession: (id) => api.post(`/api/interview/sessions/${id}/complete`),
  deleteSession: (id) => api.delete(`/api/interview/sessions/${id}`),
};

// Jobs API calls
export const jobsAPI = {
  getJobs: (skip = 0, limit = 50) => api.get(`/api/jobs?skip=${skip}&limit=${limit}`),
  getJob: (id) => api.get(`/api/jobs/${id}`),
  searchJobs: (query, limit = 50) => api.get(`/api/jobs/search?q=${encodeURIComponent(query)}&limit=${limit}`),
  getJobRecommendations: (resumeId, limit = 10) => api.get(`/api/jobs/recommendations/match?resume_id=${resumeId}&limit=${limit}`),
  getLiveJobRecommendations: (resumeId) => api.get(`/api/jobs/live/search?resume_id=${resumeId}`),
  getSavedJobs: () => api.get('/api/jobs/saved'),
  saveJob: (jobId) => api.post(`/api/jobs/saved/${jobId}`),
  unsaveJob: (jobId) => api.delete(`/api/jobs/saved/${jobId}`),
};

export const applicationsAPI = {
  getApplications: (statusFilter) => {
    const url = statusFilter ? `/api/applications?status_filter=${statusFilter}` : '/api/applications';
    return api.get(url);
  },
  getApplication: (id) => api.get(`/api/applications/${id}`),
  createApplication: (data) => api.post('/api/applications', data),
  updateApplication: (id, data) => api.put(`/api/applications/${id}`, data),
  deleteApplication: (id) => api.delete(`/api/applications/${id}`),
  getApplicationStats: () => api.get('/api/applications/stats/summary'),
};

// Analytics API calls
export const analyticsAPI = {
  getUserAnalytics: (timeRange = '30d') => api.get(`/api/analytics/user?time_range=${timeRange}`),
};

// Evaluation API calls
export const evaluationAPI = {
  runEvaluation: () => api.post('/api/evaluation/run'),
};

// Notifications API calls
export const notificationsAPI = {
  getNotifications: () => api.get('/api/notifications'),
  getUnreadCount: () => api.get('/api/notifications/unread-count'),
  markAsRead: (id) => api.put(`/api/notifications/${id}/read`),
  markAllAsRead: () => api.put('/api/notifications/read-all'),
  deleteNotification: (id) => api.delete(`/api/notifications/${id}`),
};

// Admin API calls
export const adminAPI = {
  getStats: () => api.get('/api/admin/stats'),
};

// Calendar API calls
export const calendarAPI = {
  getEvents: () => api.get('/api/calendar/events'),
  createEvent: (data) => api.post('/api/calendar/events', data),
  updateEvent: (id, data) => api.put(`/api/calendar/events/${id}`, data),
  deleteEvent: (id) => api.delete(`/api/calendar/events/${id}`),
};

export default api;
