const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
app.use(cors());
app.use(bodyParser.json());

let profiles = [];
let events = [];
let messages = [];
let votes = [0, 0, 0];
let reminders = [];

app.post('/api/profile', (req, res) => {
  profiles.push(req.body);
  res.json({ message: 'Profile created', profile: req.body });
});

app.post('/api/event', (req, res) => {
  events.push(req.body);
  res.json({ message: 'Event created', event: req.body });
});

app.post('/api/message', (req, res) => {
  messages.push(req.body.message);
  res.json({ message: 'Message sent' });
});

app.get('/api/messages', (req, res) => {
  res.json({ messages });
});

app.post('/api/vote', (req, res) => {
  const { index } = req.body;
  if (index >= 0 && index < votes.length) {
    votes[index]++;
    res.json({ message: 'Vote recorded', votes });
  } else {
    res.status(400).json({ error: 'Invalid vote index' });
  }
});

app.post('/api/reminder', (req, res) => {
  reminders.push(req.body);
  res.json({ message: 'Reminder set', reminder: req.body });
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:4000`);
});
