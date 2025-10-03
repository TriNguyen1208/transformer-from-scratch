"use client"
// import Stars from "./stars";
import Image from "next/image"

export default function BackgroundSky() {
    return (
        <div className="absolute inset-0 w-full h-screen overflow-hidden z-20">
            <Image
                src="/assets/background.png"
                alt="Night Sky"
                fill  // 👈 makes the image cover the whole container
                className="object-cover z-0"
                priority
            />
            {/* <Stars count={stars_count} /> */}
        </div>
    );
}