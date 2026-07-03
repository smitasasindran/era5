import ExperimentLayout from "../components/layout/ExperimentLayout";
import Card from "../components/common/Card";

function DepthPage() {
  return (
    <ExperimentLayout
      title="Network Depth"
      claim="Adding more layers lets a network represent increasingly complex functions of its input."
    >
      <Card title="Overview">...</Card>
      <Card title="Interactive Demo">...</Card>
      <Card title="What to Observe">...</Card>
      <Card title="Why It Matters">...</Card>
      <Card title="Summary">...</Card>
    </ExperimentLayout>
  );
}

export default DepthPage;
