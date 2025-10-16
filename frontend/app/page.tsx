// import Image from "next/image";
// import SideBar from "./components/SideBar";
import Link from "next/link";

export default function HomePage() {
    return (
        <main className="flex flex-col justify-center min-h-screen items-center bg-black text-white px-6">
            <h1 className="text-5xl font-bold">
                Chào mừng đến với <span className="text-blue-400">WeatherGPT</span>
            </h1>

            <p className="mt-4 text-lg text-gray-400">
                Mô hình AI hỏi đáp thông tin thời tiết hiện đại.
            </p>

            <Link href="/chat" className="mt-10 rounded-2xl bg-blue-500 px-6 py-3 text-lg font-semibold text-black transition hover:bg-blue-400">
                Bắt đầu ngay
            </Link>
        </main>
    );
}
