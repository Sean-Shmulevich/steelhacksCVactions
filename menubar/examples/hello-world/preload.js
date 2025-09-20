const {contextBridge, ipcRenderer} = require('electron');

contextBridge.exposeInMainWorld('electron', {
  runTerminal : async (cmd) => {
    // send the command to main, wait for result
    const result = await ipcRenderer.invoke('run-terminal', cmd);
    return result; // contains { stdout, stderr } or throws error
  }
});
