import { cp, mkdir } from 'node:fs/promises';

await mkdir('webui/dist', { recursive: true });
await cp('webui/src/index.html', 'webui/dist/index.html');
await cp('webui/src/app.js', 'webui/dist/app.js');
await cp('webui/src/app.css', 'webui/dist/app.css');
