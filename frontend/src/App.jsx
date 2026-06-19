import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Landing from './pages/Landing';
import Analyzer from './pages/Analyzer';
import Insights from './pages/Insights';
import ContentGaps from './pages/ContentGaps';
import ABTest from './pages/ABTest';
import ChannelAnalyzer from './pages/ChannelAnalyzer';
import TitleImprover from './pages/TitleImprover';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/analyze" element={<Analyzer />} />
        <Route path="/insights" element={<Insights />} />
        <Route path="/gaps" element={<ContentGaps />} />
        <Route path="/channel" element={<ChannelAnalyzer />} />
        <Route path="/improve" element={<TitleImprover />} />
        <Route path="/ab-test" element={<ABTest />} />
        <Route path="/compare" element={<Navigate to="/insights" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <Footer />
    </BrowserRouter>
  );
}
