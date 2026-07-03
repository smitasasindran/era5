import { createBrowserRouter } from "react-router-dom";

import PageLayout from "./components/layout/PageLayout";
import Home from "./pages/Home";
import ActivationPage from "./pages/ActivationPage";
import DepthPage from "./pages/DepthPage";
import EmbeddingPage from "./pages/EmbeddingPage";
import GeneralizationPage from "./pages/GeneralizationPage";

export const router = createBrowserRouter([
    {
        path: "/",
        element: <PageLayout />,
        children: [
            { index: true, element: <Home /> },
            { path: "activations", element: <ActivationPage /> },
            { path: "depth", element: <DepthPage /> },
            { path: "embeddings", element: <EmbeddingPage /> },
            { path: "generalization", element: <GeneralizationPage /> },
        ],
    },
]);
