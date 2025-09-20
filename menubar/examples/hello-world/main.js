const {menubar} = require('../..');
const path = require('path');
const {exec} = require('child_process');
const {ipcMain, electron} = require('electron');

const mb = menubar({
  browserWindow : {
    transparent : true,
    webPreferences : {
      preload : path.join(__dirname, 'preload.js'),
      contextIsolation : true,
      webSecurity : false,
      allowRunningInsecureContent : true
    }
  }
});

mb.on('ready', () => {
  console.log('Menubar app is ready.');

  // handle a channel named 'run-terminal'
  ipcMain.handle('run-terminal', async (event, cmd) => {
    return new Promise((resolve, reject) => {
      exec(cmd, (error, stdout, stderr) => {
        if (error) {
          reject({error : error.message, stderr});
        } else {
          resolve({stdout, stderr});
        }
      });
    });
  });
});
