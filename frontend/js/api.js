const API = (() => {
  const BASE = '/api';

  function teamId() {
    return localStorage.getItem('earthai_team_id') || '';
  }

  function teamName() {
    return localStorage.getItem('earthai_name') || 'Student';
  }

  async function request(method, path, body = null) {
    const opts = {
      method,
      headers: {
        'X-Team-Id': teamId(),
        'X-Team-Name': teamName(),
      },
    };
    if (body !== null) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch(`${BASE}${path}`, opts);
    const json = await resp.json();
    if (!json.ok) {
      throw new Error(json.message || json.error || 'API error');
    }
    return json.data;
  }

  return {
    get:  (path)       => request('GET', path),
    post: (path, body) => request('POST', path, body),
  };
})();
